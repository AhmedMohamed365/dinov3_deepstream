#include "optical_flow.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Middlebury color wheel (55 colors) for optical flow visualization
// Based on Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV 2007)
__constant__ float d_colorwheel[55][3] = {
    // Red to Yellow (15 steps)
    {255, 0, 0}, {255, 17, 0}, {255, 34, 0}, {255, 51, 0}, {255, 68, 0},
    {255, 85, 0}, {255, 102, 0}, {255, 119, 0}, {255, 136, 0}, {255, 153, 0},
    {255, 170, 0}, {255, 187, 0}, {255, 204, 0}, {255, 221, 0}, {255, 238, 0},
    // Yellow to Green (6 steps)
    {238, 255, 0}, {187, 255, 0}, {136, 255, 0}, {85, 255, 0}, {34, 255, 0}, {0, 255, 0},
    // Green to Cyan (4 steps)
    {0, 255, 64}, {0, 255, 128}, {0, 255, 191}, {0, 255, 255},
    // Cyan to Blue (11 steps)
    {0, 232, 255}, {0, 209, 255}, {0, 186, 255}, {0, 163, 255}, {0, 140, 255},
    {0, 116, 255}, {0, 93, 255}, {0, 70, 255}, {0, 47, 255}, {0, 23, 255}, {0, 0, 255},
    // Blue to Magenta (13 steps)
    {20, 0, 255}, {39, 0, 255}, {59, 0, 255}, {78, 0, 255}, {98, 0, 255},
    {118, 0, 255}, {137, 0, 255}, {157, 0, 255}, {176, 0, 255}, {196, 0, 255},
    {216, 0, 255}, {235, 0, 255}, {255, 0, 255},
    // Magenta to Red (6 steps)
    {255, 0, 212}, {255, 0, 170}, {255, 0, 128}, {255, 0, 85}, {255, 0, 43}, {255, 0, 0}
};

__device__ __forceinline__ float clamp_f(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ uint8_t clamp_u8(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

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

// CUDA kernel to convert optical flow to color NV12 using Middlebury color wheel
__global__ void optical_flow_to_nv12_color_kernel(
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

    // Extract flow vectors [2, H, W] layout
    int flow_offset_u = iy * flow_w + ix;
    int flow_offset_v = (flow_h * flow_w) + iy * flow_w + ix;

    float u = flow_dev[flow_offset_u];
    float v = flow_dev[flow_offset_v];

    // Compute magnitude and angle
    float rad = sqrtf(u * u + v * v);
    float angle = atan2f(-v, -u) / 3.14159265359f;  // Range: [-1, 1]

    // Normalize magnitude
    float rad_norm = fminf(rad / max_flow, 1.0f);

    // Map angle to color wheel index [0, 54]
    float fk = (angle + 1.0f) / 2.0f * 54.0f;
    int k0 = (int)floorf(fk);
    int k1 = k0 + 1;
    if (k1 == 55) k1 = 0;  // Wrap around
    float f = fk - k0;      // Interpolation factor

    // Interpolate RGB from color wheel
    float r = (1.0f - f) * d_colorwheel[k0][0] + f * d_colorwheel[k1][0];
    float g = (1.0f - f) * d_colorwheel[k0][1] + f * d_colorwheel[k1][1];
    float b = (1.0f - f) * d_colorwheel[k0][2] + f * d_colorwheel[k1][2];

    // Adjust saturation based on magnitude
    if (rad_norm <= 1.0f) {
        // Inside range: desaturate based on magnitude (less flow = whiter)
        r = (1.0f - rad_norm) * 255.0f + rad_norm * r;
        g = (1.0f - rad_norm) * 255.0f + rad_norm * g;
        b = (1.0f - rad_norm) * 255.0f + rad_norm * b;
    } else {
        // Out of range: darken (multiply by 0.75)
        r *= 0.75f;
        g *= 0.75f;
        b *= 0.75f;
    }

    // Convert RGB to YUV
    // BT.601 standard: Y = 0.299R + 0.587G + 0.114B
    //                  U = -0.169R - 0.331G + 0.5B + 128
    //                  V = 0.5R - 0.419G - 0.081B + 128
    float Y = 0.299f * r + 0.587f * g + 0.114f * b;
    float U = -0.169f * r - 0.331f * g + 0.5f * b + 128.0f;
    float V = 0.5f * r - 0.419f * g - 0.081f * b + 128.0f;

    // Write Y plane
    y_dev[oy * pitch_y + ox] = clamp_u8((int)Y);

    // Write UV plane (subsampled 2x2)
    if (ox % 2 == 0 && oy % 2 == 0) {
        int uv_x = ox;
        int uv_y = oy / 2;
        int uv_offset = uv_y * pitch_uv + uv_x;

        uv_dev[uv_offset] = clamp_u8((int)U);      // U
        uv_dev[uv_offset + 1] = clamp_u8((int)V);  // V
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

cudaError_t optical_flow_to_nv12_color_launch(
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

    optical_flow_to_nv12_color_kernel<<<grid, block, 0, stream>>>(
        flow_dev, flow_w, flow_h,
        y_dev, uv_dev,
        out_w, out_h,
        pitch_y, pitch_uv,
        max_flow);

    return cudaGetLastError();
}
