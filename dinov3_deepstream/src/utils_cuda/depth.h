#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

cudaError_t depth_to_nv12_launch(
    const float* depth_dev, int inW, int inH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH, int pitchY,
    float near_m, float far_m,
    cudaStream_t stream);