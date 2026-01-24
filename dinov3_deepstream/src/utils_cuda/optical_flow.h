#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * Visualize optical flow as grayscale intensity in NV12 format.
 * Flow magnitude maps to brightness (more flow = brighter).
 *
 * @param flow_dev Input flow tensor [2, H, W] with (u, v) flow vectors (GPU)
 * @param flow_w Flow tensor width
 * @param flow_h Flow tensor height
 * @param y_dev Output Y plane (grayscale) (GPU)
 * @param uv_dev Output UV plane (chrominance) (GPU)
 * @param out_w Output surface width
 * @param out_h Output surface height
 * @param pitch_y Y plane pitch (stride) in bytes
 * @param pitch_uv UV plane pitch (stride) in bytes
 * @param max_flow Maximum expected flow magnitude (for normalization)
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
cudaError_t optical_flow_to_nv12_grayscale_launch(
    const float* flow_dev,
    int flow_w, int flow_h,
    uint8_t* y_dev, uint8_t* uv_dev,
    int out_w, int out_h,
    int pitch_y, int pitch_uv,
    float max_flow,
    cudaStream_t stream);
