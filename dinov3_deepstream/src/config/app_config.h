#pragma once

#include <string>
#include <cstdint>

// Source type enumeration
enum class SourceType {
    CAMERA,      // USB/V4L2 camera (e.g., /dev/video0)
    FILE,        // Video file (mp4, avi, etc.)
    RTSP,        // RTSP stream
    URI          // Generic URI (http, file://, etc.)
};

// Display mode enumeration
enum class DisplayMode {
    SEPARATE,  // Each head in a separate window (default)
    TILED      // All heads tiled in a single window
};

// Pipeline configuration
struct PipelineConfig {
    SourceType source_type = SourceType::CAMERA;
    std::string source_uri = "/dev/video0";  // Camera device, file path, or RTSP URL
    DisplayMode display_mode = DisplayMode::TILED;  // Display mode for output heads
    int framerate = 30;
    int batch_size = 1;
    int width = 640;
    int height = 640;
    int live_source = 1;             // 1 for live sources (camera, RTSP), 0 for files
    int batched_push_timeout = 40000;
};

// Model configuration paths
struct ModelPathsConfig {
    std::string dinov3_config = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_dinov3.txt";
    std::string depth_config = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_depth.txt";
    std::string detection_config = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_detection.txt";
    std::string segmentation_config = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_segmentation.txt";
    std::string segmentation_labels = "/dinov3_deepstream/dinov3_models/head_segmentation/class_names.txt";
    std::string optical_flow_config = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_optical_flow.txt";
};

// Unique IDs for inference elements
struct InferenceIdsConfig {
    uint64_t dinov3_uid = 1;
    uint64_t depth_uid = 2;
    uint64_t detection_uid = 3;
    uint64_t segmentation_uid = 4;
    uint64_t optical_flow_uid = 5;
};

// Inference enable flags
struct InferenceEnableConfig {
    bool depth = true;
    bool detection = true;
    bool segmentation = true;
    bool optical_flow = true;
};

// Layer names
struct LayerNamesConfig {
    std::string features = "features";
    std::string depth = "depth";
    std::string semantic_segmentation = "semantic_segmentation";
    std::string optical_flow = "optical_flow";
};

// Visualization settings
struct VisualizationConfig {
    float alpha = 1.0f;
    int top_k = 8;
    int min_pixels = 300;

    struct FontParams {
        std::string name = "Serif";
        int size = 14;
        float color_r = 1.0f;
        float color_g = 1.0f;
        float color_b = 1.0f;
        float color_a = 1.0f;
        float bg_r = 0.0f;
        float bg_g = 0.0f;
        float bg_b = 0.0f;
        float bg_a = 0.7f;
    } font;
};

// Depth visualization range
struct DepthRangeConfig {
    float near_m = 0.5f;
    float far_m = 4.0f;
};

// Debug settings
struct DebugConfig {
    bool enabled = false;
    uint64_t initial_frames = 10;
    uint64_t periodic_interval = 120;
    std::string dot_file_path = "/dinov3_deepstream/dinov3_deepstream/build/pipeline.dot";  // Full path for DOT file (including extension)
};

// Master configuration structure
struct AppConfig {
    PipelineConfig pipeline;
    ModelPathsConfig model_paths;
    InferenceIdsConfig inference_ids;
    InferenceEnableConfig inference_enable;
    LayerNamesConfig layer_names;
    VisualizationConfig visualization;
    DepthRangeConfig depth_range;
    DebugConfig debug;

    // Factory method to create default config
    static AppConfig create_default() {
        return AppConfig{};
    }
};
