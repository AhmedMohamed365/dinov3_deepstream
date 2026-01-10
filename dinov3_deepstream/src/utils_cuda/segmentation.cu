#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t clamp_u8(int v) {
  return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__device__ __forceinline__ float clamp01(float x) {
  return fminf(1.0f, fmaxf(0.0f, x));
}

__device__ __forceinline__ void hsv_to_rgb(float h, float s, float v, float& R, float& G, float& B) {
  // h in [0,1)
  float c = v * s;
  float hh = h * 6.0f;
  float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
  float m = v - c;

  float r=0,g=0,b=0;
  if      (0.0f <= hh && hh < 1.0f) { r=c; g=x; b=0; }
  else if (1.0f <= hh && hh < 2.0f) { r=x; g=c; b=0; }
  else if (2.0f <= hh && hh < 3.0f) { r=0; g=c; b=x; }
  else if (3.0f <= hh && hh < 4.0f) { r=0; g=x; b=c; }
  else if (4.0f <= hh && hh < 5.0f) { r=x; g=0; b=c; }
  else                               { r=c; g=0; b=x; }

  R = (r + m);
  G = (g + m);
  B = (b + m);
}

__device__ __forceinline__ void rgb_to_yuv_u8(float R, float G, float B, uint8_t& Y, uint8_t& U, uint8_t& V) {
  // R,G,B in [0,1]
  float r = 255.0f * R;
  float g = 255.0f * G;
  float b = 255.0f * B;

  float y =  0.299f * r + 0.587f * g + 0.114f * b;
  float u = -0.169f * r - 0.331f * g + 0.500f * b + 128.0f;
  float v =  0.500f * r - 0.419f * g - 0.081f * b + 128.0f;

  Y = clamp_u8((int)(y + 0.5f));
  U = clamp_u8((int)(u + 0.5f));
  V = clamp_u8((int)(v + 0.5f));
}

template<typename T>
__device__ __forceinline__ float load_t(const T* p);

template<>
__device__ __forceinline__ float load_t<float>(const float* p) { return *p; }

template<>
__device__ __forceinline__ float load_t<__half>(const __half* p) { return __half2float(*p); }

template<typename T>
__device__ __forceinline__ int argmax_c(const T* logits, int C, int W, int H, int x, int y) {
  // logits layout assumed CHW (C, H, W) contiguous
  int idx = y * W + x;
  float best = -1e30f;
  int best_c = 0;
  int stride = W * H;
  for (int c = 0; c < C; ++c) {
    float v = load_t<T>(logits + c * stride + idx);
    if (v > best) { best = v; best_c = c; }
  }
  return best_c;
}

template<typename T>
__global__ void seg_to_nv12_y_kernel(
    const T* __restrict__ logits, int C, int inW, int inH,
    uint8_t* __restrict__ y_plane, int outW, int outH, int pitchY)
{
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= outW || oy >= outH) return;

  int sx = (ox * inW) / outW;
  int sy = (oy * inH) / outH;

  int cls = argmax_c<T>(logits, C, inW, inH, sx, sy);

  float R=0,G=0,B=0;
  if (cls != 0) {
    // golden-ratio-ish hue spacing for distinct colors
    float h = fmodf(cls * 0.61803398875f, 1.0f);
    hsv_to_rgb(h, 1.0f, 1.0f, R, G, B);
  }

  uint8_t Y,U,V;
  rgb_to_yuv_u8(R, G, B, Y, U, V);
  y_plane[oy * pitchY + ox] = Y;
}

template<typename T>
__global__ void seg_to_nv12_uv_kernel(
    const T* __restrict__ logits, int C, int inW, int inH,
    uint8_t* __restrict__ uv_plane, int outW, int outH, int pitchUV)
{
  // chroma samples: outW/2 by outH/2, each writes 2 bytes (U,V) at x=2*cx
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;
  int uvW = outW >> 1;
  int uvH = outH >> 1;
  if (cx >= uvW || cy >= uvH) return;

  int ox = cx << 1;
  int oy = cy << 1;

  uint32_t sumU = 0, sumV = 0;
  int count = 0;

  // average chroma from the 2x2 block
  #pragma unroll
  for (int dy = 0; dy < 2; ++dy) {
    int oy2 = oy + dy;
    if (oy2 >= outH) continue;
    int sy = (oy2 * inH) / outH;

    #pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
      int ox2 = ox + dx;
      if (ox2 >= outW) continue;
      int sx = (ox2 * inW) / outW;

      int cls = argmax_c<T>(logits, C, inW, inH, sx, sy);

      float R=0,G=0,B=0;
      if (cls != 0) {
        float h = fmodf(cls * 0.61803398875f, 1.0f);
        hsv_to_rgb(h, 1.0f, 1.0f, R, G, B);
      }

      uint8_t Y,U,V;
      rgb_to_yuv_u8(R, G, B, Y, U, V);
      sumU += (uint32_t)U;
      sumV += (uint32_t)V;
      count++;
    }
  }

  uint8_t Uavg = (count > 0) ? (uint8_t)((sumU + (count/2)) / count) : 128;
  uint8_t Vavg = (count > 0) ? (uint8_t)((sumV + (count/2)) / count) : 128;

  int byte_x = ox; // U at [byte_x], V at [byte_x+1]
  uv_plane[cy * pitchUV + byte_x]     = Uavg;
  uv_plane[cy * pitchUV + byte_x + 1] = Vavg;
}

extern "C"
cudaError_t seg_logits_to_nv12_launch(
    const void* logits_dev, bool is_half,
    int C, int inW, int inH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH, int pitchY, int pitchUV,
    cudaStream_t stream)
{
  dim3 block(16,16);
  dim3 grid((outW + block.x - 1) / block.x,
            (outH + block.y - 1) / block.y);

  dim3 blockUV(16,16);
  dim3 gridUV(((outW/2) + blockUV.x - 1) / blockUV.x,
              ((outH/2) + blockUV.y - 1) / blockUV.y);

  if (is_half) {
    auto* p = (const __half*)logits_dev;
    seg_to_nv12_y_kernel<__half><<<grid, block, 0, stream>>>(p, C, inW, inH, y_dev, outW, outH, pitchY);
    seg_to_nv12_uv_kernel<__half><<<gridUV, blockUV, 0, stream>>>(p, C, inW, inH, uv_dev, outW, outH, pitchUV);
  } else {
    auto* p = (const float*)logits_dev;
    seg_to_nv12_y_kernel<float><<<grid, block, 0, stream>>>(p, C, inW, inH, y_dev, outW, outH, pitchY);
    seg_to_nv12_uv_kernel<float><<<gridUV, blockUV, 0, stream>>>(p, C, inW, inH, uv_dev, outW, outH, pitchUV);
  }

  // no sync here by design
  return cudaPeekAtLastError();
}


// /////////////////////////////////////////////////////////////
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