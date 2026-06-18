#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

cudaError_t seg_argmax_launch(
    const void* logits_dev,  // float* or __half*
    bool is_half,
    int C, int H, int W,      // logits shape: [1,C,H,W]
    int32_t* class_map_dev,   // output: [H,W] int32 (class id)
    cudaStream_t stream);
    
cudaError_t seg_classmap_to_nv12_launch(
    const int32_t* class_map_dev, int mapW, int mapH,
    uint8_t* y_dev, uint8_t* uv_dev,
    int outW, int outH, int pitchY, int pitchUV,
    float alpha,              // 1.0 = pure color, 0.0 = keep original
    cudaStream_t stream);

    cudaError_t accumulate_centroids_kernel_launch(
        const int32_t* class_map_dev,
        int W, int H,
        int num_classes,
        int32_t* count_dev,
        int64_t* sumx_dev,
        int64_t* sumy_dev,
        cudaStream_t stream);

struct TranslationVector {
    int class_id;
    int dx;
    int dy;
};

cudaError_t translate_segmentation_masks_launch(
    const int32_t* src_class_map,
    int32_t* dst_class_map,
    int W, int H,
    const TranslationVector* translations_dev,
    int num_translations,
    cudaStream_t stream);