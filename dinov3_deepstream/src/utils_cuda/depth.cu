#include <cuda_runtime.h>
#include <stdint.h>

// __device__ __forceinline__ uint8_t clamp_u8(int v) {
//   return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
// }

// // Writes grayscale into Y plane, nearest-neighbor resize (inW,inH)->(outW,outH)
// __global__ void depth_to_nv12_y_kernel(
//     const float* __restrict__ depth, int inW, int inH,
//     uint8_t* __restrict__ y, int outW, int outH, int pitchY,
//     float near_m, float far_m)
// {
//   int ox = blockIdx.x * blockDim.x + threadIdx.x;
//   int oy = blockIdx.y * blockDim.y + threadIdx.y;
//   if (ox >= outW || oy >= outH) return;

//   int sx = (ox * inW) / outW;
//   int sy = (oy * inH) / outH;

//   float d = depth[sy * inW + sx]; // absolute depth in meters

//   // Map depth range [near_m..far_m] to [255..0] (near=bright, far=dark). Flip if you want.
//   float t = (d - near_m) / (far_m - near_m);
//   if (t < 0.f) t = 0.f;
//   if (t > 1.f) t = 1.f;

//   int g = (int)((1.0f - t) * 255.0f + 0.5f);
//   y[oy * pitchY + ox] = clamp_u8(g);
// }

// // Set UV plane to 128 (neutral chroma) -> grayscale
// __global__ void nv12_set_uv_kernel(uint8_t* __restrict__ uv, int uvW, int uvH, int pitchUV)
// {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   if (x >= uvW || y >= uvH) return;

//   // NV12 UV plane is interleaved bytes: U,V,U,V...
//   uv[y * pitchUV + x] = 128;
// }


// // C-callable wrapper (so main.cpp can call it)
// cudaError_t depth_to_nv12_launch(
//     const float* depth_dev, int inW, int inH,
//     uint8_t* y_dev, uint8_t* uv_dev,
//     int outW, int outH, int pitchY,
//     float near_m, float far_m,
//     cudaStream_t stream)
// {
//   dim3 block(16, 16);
//   dim3 grid((outW + block.x - 1) / block.x,
//             (outH + block.y - 1) / block.y);

//   depth_to_nv12_y_kernel<<<grid, block, 0, stream>>>(
//       depth_dev, inW, inH, y_dev, outW, outH, pitchY, near_m, far_m);

//   int uvH = outH / 2;
//   dim3 block2(32, 8);
//   dim3 grid2((outW + block2.x - 1) / block2.x,
//              (uvH + block2.y - 1) / block2.y);

//   nv12_set_uv_kernel<<<grid2, block2, 0, stream>>>(uv_dev, outW, uvH, pitchY);

//   return cudaGetLastError();
// }


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

// Simple "inferno-ish" colormap using a few control points + linear interpolation.
// t=0 -> far (dark), t=1 -> near (bright)
__device__ __forceinline__ uchar3 colormap_infernoish(float t) {
  // Control points (RGB)
  // 0.00: near-black
  // 0.25: purple
  // 0.50: magenta/red
  // 0.75: orange
  // 1.00: yellow-white
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

// RGB->YUV (BT.601 full-range-ish). Good enough for visualization.
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

__global__ void depth_to_nv12_colormap_kernel(
    const float* __restrict__ depth, int inW, int inH,
    uint8_t* __restrict__ y_plane,
    uint8_t* __restrict__ uv_plane,
    int outW, int outH,
    int pitchY, int pitchUV,
    float dmin, float dmax)
{
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= outW || oy >= outH) return;

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

  // Write UV for top-left of each 2x2 block (NV12 subsampling)
  // UV plane has size outW x (outH/2), interleaved U,V for each 2 pixels.
  if (((ox & 1) == 0) && ((oy & 1) == 0)) {
    int uvY = oy >> 1;
    int uvX = ox; // points to U; V at uvX+1

    // Guard if outW is odd (usually it's even for NV12)
    if (uvX + 1 < outW && uvY < (outH >> 1)) {

      // Average chroma over the 2x2 block for better color
      // Sample the 4 pixels: (ox,oy), (ox+1,oy), (ox,oy+1), (ox+1,oy+1)
      // Each has its own mapped depth sample.
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

cudaError_t depth_to_nv12_colormap_launch(
    const float* depth_dev, int inW, int inH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH,
    int pitchY, int pitchUV,
    float dmin, float dmax,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((outW + block.x - 1) / block.x,
            (outH + block.y - 1) / block.y);

  depth_to_nv12_colormap_kernel<<<grid, block, 0, stream>>>(
      depth_dev, inW, inH,
      y_dev, uv_dev,
      outW, outH,
      pitchY, pitchUV,
      dmin, dmax);

  return cudaGetLastError();
}
