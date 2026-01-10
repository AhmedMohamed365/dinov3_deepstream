#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void argmax_float_kernel(
  const float* __restrict__ logits,
  int C, int H, int W,
  int32_t* __restrict__ out)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= W || y >= H) return;

int idx_hw = y * W + x;
int base = idx_hw; // c*H*W + idx_hw

float best = logits[base];
int best_c = 0;

int stride = H * W;
for (int c = 1; c < C; ++c) {
  float v = logits[c * stride + idx_hw];
  if (v > best) { best = v; best_c = c; }
}

out[idx_hw] = best_c;
}

__global__ void argmax_half_kernel(
  const __half* __restrict__ logits,
  int C, int H, int W,
  int32_t* __restrict__ out)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= W || y >= H) return;

int idx_hw = y * W + x;
int stride = H * W;

float best = __half2float(logits[idx_hw]);
int best_c = 0;

for (int c = 1; c < C; ++c) {
  float v = __half2float(logits[c * stride + idx_hw]);
  if (v > best) { best = v; best_c = c; }
}

out[idx_hw] = best_c;
}

cudaError_t seg_argmax_launch(
  const void* logits_dev,
  bool is_half,
  int C, int H, int W,
  int32_t* class_map_dev,
  cudaStream_t stream)
{
dim3 block(16, 16);
dim3 grid((W + block.x - 1) / block.x,
          (H + block.y - 1) / block.y);

if (is_half) {
  argmax_half_kernel<<<grid, block, 0, stream>>>(
      (const __half*)logits_dev, C, H, W, class_map_dev);
} else {
  argmax_float_kernel<<<grid, block, 0, stream>>>(
      (const float*)logits_dev, C, H, W, class_map_dev);
}

return cudaGetLastError();
}





static __device__ __forceinline__ uint8_t clamp_u8(int v) {
  return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static __device__ __forceinline__ uchar3 color_from_id(int id) {
  // background black
  if (id <= 0) return make_uchar3(0,0,0);
  // person red
  if (id == 1) return make_uchar3(255,0,0);

  // Deterministic hash -> RGB
  uint32_t x = (uint32_t)id * 2654435761u;
  uint8_t r = (x >> 16) & 0xFF;
  uint8_t g = (x >>  8) & 0xFF;
  uint8_t b = (x >>  0) & 0xFF;

  // Avoid too-dark colors
  r = (uint8_t)(64 + (r >> 1));
  g = (uint8_t)(64 + (g >> 1));
  b = (uint8_t)(64 + (b >> 1));
  return make_uchar3(r,g,b);
}

// BT.601-ish conversion (good enough for visualization)
static __device__ __forceinline__ uint8_t rgb_to_y(uchar3 c) {
  int y = (  66 * c.x + 129 * c.y +  25 * c.z + 128) >> 8;
  y += 16;
  return clamp_u8(y);
}
static __device__ __forceinline__ uint8_t rgb_to_u(uchar3 c) {
  int u = ( -38 * c.x -  74 * c.y + 112 * c.z + 128) >> 8;
  u += 128;
  return clamp_u8(u);
}
static __device__ __forceinline__ uint8_t rgb_to_v(uchar3 c) {
  int v = ( 112 * c.x -  94 * c.y -  18 * c.z + 128) >> 8;
  v += 128;
  return clamp_u8(v);
}

__global__ void classmap_to_nv12_y(
    const int32_t* __restrict__ cls, int mapW, int mapH,
    uint8_t* __restrict__ y, int outW, int outH, int pitchY,
    float alpha)
{
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int ypix = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= outW || ypix >= outH) return;

  int sx = (int)((int64_t)x * mapW / outW);
  int sy = (int)((int64_t)ypix * mapH / outH);
  int id = cls[sy * mapW + sx];

  uchar3 rgb = color_from_id(id);
  uint8_t y_col = rgb_to_y(rgb);

  uint8_t* row = y + (size_t)ypix * pitchY;
  if (alpha >= 1.0f) {
    row[x] = y_col;
  } else {
    float yin = (float)row[x];
    row[x] = (uint8_t)(alpha * (float)y_col + (1.0f - alpha) * yin);
  }
}

__global__ void classmap_to_nv12_uv(
    const int32_t* __restrict__ cls, int mapW, int mapH,
    uint8_t* __restrict__ uv, int outW, int outH, int pitchUV,
    float alpha)
{
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x); // uv x
  int yuv = (int)(blockIdx.y * blockDim.y + threadIdx.y); // uv y
  int uvW = outW >> 1;
  int uvH = outH >> 1;
  if (x >= uvW || yuv >= uvH) return;

  // Corresponding top-left pixel of the 2x2 block
  int px = x << 1;
  int py = yuv << 1;

  // Sample one class id (fast). If you want nicer chroma, average 4 samples.
  int sx = (int)((int64_t)px * mapW / outW);
  int sy = (int)((int64_t)py * mapH / outH);
  int id = cls[sy * mapW + sx];

  uchar3 rgb = color_from_id(id);
  uint8_t u_col = rgb_to_u(rgb);
  uint8_t v_col = rgb_to_v(rgb);

  uint8_t* row = uv + (size_t)yuv * pitchUV;
  int o = x * 2;

  if (alpha >= 1.0f) {
    row[o + 0] = u_col;
    row[o + 1] = v_col;
  } else {
    float uin = (float)row[o + 0];
    float vin = (float)row[o + 1];
    row[o + 0] = (uint8_t)(alpha * (float)u_col + (1.0f - alpha) * uin);
    row[o + 1] = (uint8_t)(alpha * (float)v_col + (1.0f - alpha) * vin);
  }
}

cudaError_t seg_classmap_to_nv12_launch(
    const int32_t* class_map_dev, int mapW, int mapH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH, int pitchY, int pitchUV,
    float alpha,
    cudaStream_t stream)
{
  dim3 block(16, 16);

  dim3 gridY((outW + block.x - 1) / block.x,
             (outH + block.y - 1) / block.y);
  classmap_to_nv12_y<<<gridY, block, 0, stream>>>(
      class_map_dev, mapW, mapH,
      y_dev, outW, outH, pitchY, alpha);

  int uvW = outW >> 1, uvH = outH >> 1;
  dim3 gridUV((uvW + block.x - 1) / block.x,
              (uvH + block.y - 1) / block.y);
  classmap_to_nv12_uv<<<gridUV, block, 0, stream>>>(
      class_map_dev, mapW, mapH,
      uv_dev, outW, outH, pitchUV, alpha);

  return cudaGetLastError();
}