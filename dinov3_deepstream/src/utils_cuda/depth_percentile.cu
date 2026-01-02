#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>

// ------------------------ small helpers ------------------------

__device__ __forceinline__ uint8_t clamp_u8(int v) {
  return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__device__ __forceinline__ float clamp01(float x) {
  return fminf(1.0f, fmaxf(0.0f, x));
}

__device__ __forceinline__ bool is_valid_depth(float d) {
  // d == d filters NaN; also require positive
  return (d == d) && (d > 0.0f);
}

// Atomic min/max for float using atomicCAS.
// Assumes inputs are finite (we filter NaN before calling).
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
  int* addr_i = (int*)addr;
  int old = *addr_i;
  while (__int_as_float(old) > value) {
    int assumed = old;
    old = atomicCAS(addr_i, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  int* addr_i = (int*)addr;
  int old = *addr_i;
  while (__int_as_float(old) < value) {
    int assumed = old;
    old = atomicCAS(addr_i, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

// ------------------------ colormap + RGB->YUV ------------------------

// Simple "inferno-ish" colormap using control points + linear interpolation.
// t=0 -> far (dark), t=1 -> near (bright)
__device__ __forceinline__ uchar3 colormap_infernoish(float t) {
  const float3 c0 = make_float3(0.f,   0.f,   0.f);
  const float3 c1 = make_float3(60.f,  0.f, 110.f);
  const float3 c2 = make_float3(180.f, 30.f, 80.f);
  const float3 c3 = make_float3(250.f, 140.f, 20.f);
  const float3 c4 = make_float3(255.f, 255.f, 180.f);

  t = clamp01(t);

  float3 a, b;
  float u;

  if (t < 0.25f) {
    a = c0; b = c1; u = t / 0.25f;
  } else if (t < 0.50f) {
    a = c1; b = c2; u = (t - 0.25f) / 0.25f;
  } else if (t < 0.75f) {
    a = c2; b = c3; u = (t - 0.50f) / 0.25f;
  } else {
    a = c3; b = c4; u = (t - 0.75f) / 0.25f;
  }

  float3 c;
  c.x = a.x + (b.x - a.x) * u;
  c.y = a.y + (b.y - a.y) * u;
  c.z = a.z + (b.z - a.z) * u;

  return make_uchar3((uint8_t)(c.x + 0.5f),
                     (uint8_t)(c.y + 0.5f),
                     (uint8_t)(c.z + 0.5f));
}

// RGB->YUV (BT.601-ish). Good enough for visualization.
__device__ __forceinline__ void rgb_to_yuv_u8(uchar3 rgb, uint8_t& Y, uint8_t& U, uint8_t& V) {
  float R = (float)rgb.x;
  float G = (float)rgb.y;
  float B = (float)rgb.z;

  float y =  0.299f * R + 0.587f * G + 0.114f * B;
  float u = -0.169f * R - 0.331f * G + 0.500f * B + 128.0f;
  float v =  0.500f * R - 0.419f * G - 0.081f * B + 128.0f;

  Y = clamp_u8((int)(y + 0.5f));
  U = clamp_u8((int)(u + 0.5f));
  V = clamp_u8((int)(v + 0.5f));
}

// ------------------------ percentile (histogram) pipeline ------------------------

static constexpr int BINS = 2048;

__global__ void init_percentile_ws(uint32_t* hist, int bins, float2* minmax, uint32_t* valid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < bins) hist[i] = 0;
  if (i == 0) {
    *valid = 0;
    // init min=+inf, max=0
    minmax->x = 1e10f;
    minmax->y = 0.f;
  }
}

__global__ void reduce_minmax_valid(const float* depth, int N, float2* minmax, uint32_t* valid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float d = depth[i];
  if (!is_valid_depth(d)) return;

  atomicAdd(valid, 1u);
  atomicMinFloat(&minmax->x, d);
  atomicMaxFloat(&minmax->y, d);
}

__global__ void hist_depth_dev(const float* depth, int N,
                              const float2* __restrict__ minmax_dev,
                              uint32_t* hist, int bins)
{
  float mn = minmax_dev->x;
  float mx = minmax_dev->y;
  float denom = mx - mn;
  if (denom <= 1e-6f) return;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float d = depth[i];
  if (!is_valid_depth(d)) return;

  float u = (d - mn) / denom;        // 0..1
  u = clamp01(u);
  int b = (int)(u * (bins - 1));
  atomicAdd(&hist[b], 1u);
}

// Single-kernel percentile extraction: one thread walks the histogram (bins is small).
__global__ void percentiles_from_hist_dev(const uint32_t* hist, int bins,
                                         const float2* __restrict__ minmax_dev,
                                         const uint32_t* __restrict__ valid_dev,
                                         float2* out_dminmax)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  uint32_t valid = *valid_dev;
  if (valid == 0) {
    // Fallback if nothing valid
    out_dminmax->x = 0.1f;
    out_dminmax->y = 1.0f;
    return;
  }

  // Targets: first bin where cumulative >= 5% and 95%
  uint32_t t5  = (uint32_t)(0.05f * (float)valid);
  uint32_t t95 = (uint32_t)(0.95f * (float)valid);

  // Ensure non-degenerate thresholds
  if (t5 < 1) t5 = 1;
  if (t95 < t5 + 1) t95 = t5 + 1;
  if (t95 > valid) t95 = valid;

  uint32_t c = 0;
  int b5 = 0;
  for (int b = 0; b < bins; ++b) {
    c += hist[b];
    if (c >= t5) { b5 = b; break; }
  }

  c = 0;
  int b95 = bins - 1;
  for (int b = 0; b < bins; ++b) {
    c += hist[b];
    if (c >= t95) { b95 = b; break; }
  }

  float mn = minmax_dev->x;
  float mx = minmax_dev->y;
  float denom = mx - mn;
  if (denom <= 1e-6f) denom = 1e-6f;

  // Bin centers -> depth values
  float dmin = mn + ((b5  + 0.5f) / (float)bins) * denom;
  float dmax = mn + ((b95 + 0.5f) / (float)bins) * denom;

  if (dmax <= dmin) dmax = dmin + 1e-6f;

  out_dminmax->x = dmin;
  out_dminmax->y = dmax;
}

// ------------------------ render kernel (NV12 colormap) ------------------------

__global__ void depth_to_nv12_colormap_kernel(
    const float* __restrict__ depth, int inW, int inH,
    uint8_t* __restrict__ y_plane,
    uint8_t* __restrict__ uv_plane,
    int outW, int outH,
    int pitchY, int pitchUV,
    const float2* __restrict__ dminmax_dev)   // {dmin,dmax} in device memory
{
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= outW || oy >= outH) return;

  float dmin = dminmax_dev->x;
  float dmax = dminmax_dev->y;

  // Nearest-neighbor map output pixel -> depth pixel
  int sx = (ox * inW) / outW;
  int sy = (oy * inH) / outH;

  float d = depth[sy * inW + sx];
  bool valid = is_valid_depth(d);

  // Normalize: near brighter => t=1 at dmin, t=0 at dmax
  float denom = (dmax - dmin);
  float t = 0.0f;
  if (valid && denom > 1e-6f) {
    t = 1.0f - (d - dmin) / denom;
    t = clamp01(t);
  }

  uchar3 rgb = valid ? colormap_infernoish(t) : make_uchar3(0, 0, 0);

  uint8_t Y, U, V;
  rgb_to_yuv_u8(rgb, Y, U, V);

  // Write Y for every pixel
  y_plane[oy * pitchY + ox] = Y;

  // NV12 chroma: one UV sample per 2x2 luma block (subsampling)
  if (((ox & 1) == 0) && ((oy & 1) == 0)) {
    int uvY = oy >> 1;
    int uvX = ox; // points to U; V at uvX+1

    if (uvX + 1 < outW && uvY < (outH >> 1)) {
      // Average chroma over the 2x2 block
      uint32_t sumU = 0, sumV = 0;
      int count = 0;

      for (int dy = 0; dy < 2; ++dy) {
        int oy2 = oy + dy;
        if (oy2 >= outH) continue;
        int sy2 = (oy2 * inH) / outH;

        for (int dx = 0; dx < 2; ++dx) {
          int ox2 = ox + dx;
          if (ox2 >= outW) continue;
          int sx2 = (ox2 * inW) / outW;

          float d2 = depth[sy2 * inW + sx2];
          bool v2 = is_valid_depth(d2);

          float t2 = 0.0f;
          if (v2 && denom > 1e-6f) {
            t2 = 1.0f - (d2 - dmin) / denom;
            t2 = clamp01(t2);
          }

          uchar3 rgb2 = v2 ? colormap_infernoish(t2) : make_uchar3(0, 0, 0);
          uint8_t Y2, U2, V2;
          rgb_to_yuv_u8(rgb2, Y2, U2, V2);

          sumU += (uint32_t)U2;
          sumV += (uint32_t)V2;
          count++;
        }
      }

      uint8_t Uavg = (count > 0) ? (uint8_t)((sumU + (count/2)) / count) : 128;
      uint8_t Vavg = (count > 0) ? (uint8_t)((sumV + (count/2)) / count) : 128;

      uv_plane[uvY * pitchUV + uvX]     = Uavg;
      uv_plane[uvY * pitchUV + uvX + 1] = Vavg;
    }
  }
}

// ------------------------ public launcher ------------------------

cudaError_t depth_to_nv12_colormap_percentile_launch(
    const float* depth_dev, int inW, int inH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH,
    int pitchY, int pitchUV,
    cudaStream_t stream)
{
  // Persistent workspace (one allocation per process)
  static uint32_t* d_hist = nullptr;
  static float2*   d_minmax = nullptr;
  static float2*   d_dminmax = nullptr;
  static uint32_t* d_valid = nullptr;

  if (!d_hist) {
    cudaError_t e;
    e = cudaMalloc(&d_hist,   BINS * sizeof(uint32_t)); if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_minmax, sizeof(float2));          if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_dminmax,sizeof(float2));          if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_valid,  sizeof(uint32_t));        if (e != cudaSuccess) return e;
  }

  int N = inW * inH;

  // 1) init histogram + counters
  {
    int threads = 256;
    int blocks = (BINS + threads - 1) / threads;
    init_percentile_ws<<<blocks, threads, 0, stream>>>(d_hist, BINS, d_minmax, d_valid);
  }

  // 2) compute min/max + valid count
  {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reduce_minmax_valid<<<blocks, threads, 0, stream>>>(depth_dev, N, d_minmax, d_valid);
  }

  // 3) histogram (uses min/max from device memory)
  {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hist_depth_dev<<<blocks, threads, 0, stream>>>(depth_dev, N, d_minmax, d_hist, BINS);
  }

  // 4) compute p5/p95 from histogram (device-only)
  {
    percentiles_from_hist_dev<<<1, 32, 0, stream>>>(d_hist, BINS, d_minmax, d_valid, d_dminmax);
  }
}

  // 5) render NV12 using colormap and device dmin/dmax
  {
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y);

    depth_to_nv12_colormap_kernel<<<grid, block, 0, stream>>>(
        depth_dev, inW, inH,
        y_dev, uv_dev,
        outW, outH,
        pitchY, pitchUV,
        d_dminmax);
  }

  return cudaGetLastError();
}