#include "optical_flow.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// CUDA kernel to convert optical flow to grayscale NV12
__global__ void optical_flow_to_nv12_grayscale_kernel(
    const float* flow_dev,     // [2, H, W] - (u, v) channels
    int flow_w, int flow_h,
    uint8_t* y_dev,
    uint8_t* uv_dev,
    int out_w, int out_h,
    int pitch_y, int pitch_uv,
    float max_flow)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= out_w || oy >= out_h) return;

    // Map output coordinates to flow tensor coordinates
    float fx = (float)ox * flow_w / out_w;
    float fy = (float)oy * flow_h / out_h;

    int ix = (int)fx;
    int iy = (int)fy;

    if (ix >= flow_w) ix = flow_w - 1;
    if (iy >= flow_h) iy = flow_h - 1;

    // Flow tensor layout: [2, H, W] - channel 0 = u, channel 1 = v
    int flow_offset_u = iy * flow_w + ix;                    // u channel
    int flow_offset_v = (flow_h * flow_w) + iy * flow_w + ix;  // v channel

    float u = flow_dev[flow_offset_u];
    float v = flow_dev[flow_offset_v];

    // Compute flow magnitude
    float magnitude = sqrtf(u * u + v * v);

    // Normalize to [0, 1] using max_flow
    float normalized = fminf(magnitude / max_flow, 1.0f);

    // Map to grayscale intensity [0, 255]
    // More flow = brighter (255 = white, 0 = black)
    uint8_t intensity = (uint8_t)(normalized * 255.0f);

    // Write to Y plane (grayscale)
    y_dev[oy * pitch_y + ox] = intensity;

    // Write to UV plane (neutral gray - no color)
    // NV12 UV plane is subsampled 2x2, so only write for even coordinates
    if (ox % 2 == 0 && oy % 2 == 0) {
        int uv_x = ox;
        int uv_y = oy / 2;
        int uv_offset = uv_y * pitch_uv + uv_x;

        // Neutral chrominance (128, 128) = grayscale
        uv_dev[uv_offset] = 128;      // U
        uv_dev[uv_offset + 1] = 128;  // V
    }
}

cudaError_t optical_flow_to_nv12_grayscale_launch(
    const float* flow_dev,
    int flow_w, int flow_h,
    uint8_t* y_dev, uint8_t* uv_dev,
    int out_w, int out_h,
    int pitch_y, int pitch_uv,
    float max_flow,
    cudaStream_t stream)
{
    if (!flow_dev || !y_dev || !uv_dev) return cudaErrorInvalidValue;
    if (flow_w <= 0 || flow_h <= 0 || out_w <= 0 || out_h <= 0) return cudaErrorInvalidValue;

    // Use 16x16 thread blocks
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y);

    optical_flow_to_nv12_grayscale_kernel<<<grid, block, 0, stream>>>(
        flow_dev, flow_w, flow_h,
        y_dev, uv_dev,
        out_w, out_h,
        pitch_y, pitch_uv,
        max_flow);

    return cudaGetLastError();
}
