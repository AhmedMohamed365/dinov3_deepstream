#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"

// SINK pad probe - Concatenates two consecutive DINOv3 features before inference
class OpticalFlowPreprocessHandler {
public:
    OpticalFlowPreprocessHandler(const AppConfig& cfg)
        : config(cfg), frame_count(0),
          prev_features_dev(nullptr), prev_features_size(0),
          concat_features_dev(nullptr), concat_features_size(0) {}

    ~OpticalFlowPreprocessHandler();

    GstPadProbeReturn handle_buffer(GstPad* pad, GstPadProbeInfo* info);

private:
    const AppConfig& config;
    uint64_t frame_count;

    // Buffer for previous frame's DINOv3 features [C, H, W]
    void* prev_features_dev;
    size_t prev_features_size;
    int prev_H, prev_W, prev_C;

    // Buffer for concatenated features [2*C, H, W]
    void* concat_features_dev;
    size_t concat_features_size;

    struct FeatureTensorInfo {
        const void* features_dev;
        int H, W, C;
    };

    bool should_debug() const;

    bool extract_dinov3_features(
        NvDsBatchMeta* batch_meta,
        FeatureTensorInfo& out_info,
        bool debug);

    void buffer_features(const FeatureTensorInfo& features);

    bool concatenate_and_update_meta(
        NvDsBatchMeta* batch_meta,
        const FeatureTensorInfo& current_features,
        bool debug);
};

// SRC pad probe - Visualizes optical flow output after inference
class OpticalFlowVisualizationHandler {
public:
    OpticalFlowVisualizationHandler(const AppConfig& cfg)
        : config(cfg), frame_count(0) {}

    GstPadProbeReturn handle_buffer(GstPad* pad, GstPadProbeInfo* info);

private:
    const AppConfig& config;
    uint64_t frame_count;

    struct OpticalFlowTensorInfo {
        const float* flow_dev;  // [2, H, W] - (u, v) flow vectors
        int width;
        int height;
    };

    bool should_debug() const;
    bool ensure_buffer_writable(GstPadProbeInfo* info);
    NvBufSurface* map_buffer(GstBuffer* buf, GstMapInfo& map_info);
    bool validate_surface(NvBufSurface* surface, bool debug);

    bool extract_optical_flow_tensor(
        NvDsBatchMeta* batch_meta,
        NvDsFrameMeta* frame_meta,
        GstNvDsPreProcessBatchMeta* pbm,
        OpticalFlowTensorInfo& out_info,
        bool debug);

    bool process_frame_optical_flow(
        NvBufSurface* surface,
        NvDsFrameMeta* frame_meta,
        const OpticalFlowTensorInfo& flow_info,
        bool debug);
};

// C-style callback wrappers for GStreamer
GstPadProbeReturn optical_flow_sink_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data);

GstPadProbeReturn optical_flow_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data);
