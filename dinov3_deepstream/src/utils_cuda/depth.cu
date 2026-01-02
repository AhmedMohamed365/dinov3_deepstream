#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t clamp_u8(int v) {
  return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// Writes grayscale into Y plane, nearest-neighbor resize (inW,inH)->(outW,outH)
__global__ void depth_to_nv12_y_kernel(
    const float* __restrict__ depth, int inW, int inH,
    uint8_t* __restrict__ y, int outW, int outH, int pitchY,
    float near_m, float far_m)
{
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= outW || oy >= outH) return;

  int sx = (ox * inW) / outW;
  int sy = (oy * inH) / outH;

  float d = depth[sy * inW + sx]; // absolute depth in meters

  // Map depth range [near_m..far_m] to [255..0] (near=bright, far=dark). Flip if you want.
  float t = (d - near_m) / (far_m - near_m);
  if (t < 0.f) t = 0.f;
  if (t > 1.f) t = 1.f;

  int g = (int)((1.0f - t) * 255.0f + 0.5f);
  y[oy * pitchY + ox] = clamp_u8(g);
}

// Set UV plane to 128 (neutral chroma) -> grayscale
__global__ void nv12_set_uv_kernel(uint8_t* __restrict__ uv, int uvW, int uvH, int pitchUV)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= uvW || y >= uvH) return;

  // NV12 UV plane is interleaved bytes: U,V,U,V...
  uv[y * pitchUV + x] = 128;
}


// C-callable wrapper (so main.cpp can call it)
cudaError_t depth_to_nv12_launch(
    const float* depth_dev, int inW, int inH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH, int pitchY,
    float near_m, float far_m,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((outW + block.x - 1) / block.x,
            (outH + block.y - 1) / block.y);

  depth_to_nv12_y_kernel<<<grid, block, 0, stream>>>(
      depth_dev, inW, inH, y_dev, outW, outH, pitchY, near_m, far_m);

  int uvH = outH / 2;
  dim3 block2(32, 8);
  dim3 grid2((outW + block2.x - 1) / block2.x,
             (uvH + block2.y - 1) / block2.y);

  nv12_set_uv_kernel<<<grid2, block2, 0, stream>>>(uv_dev, outW, uvH, pitchY);

  return cudaGetLastError();
}
