#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"

// DINOv3 backbone probe handler
// Extracts backbone features and forwards them to downstream heads
class DINOv3ProbeHandler {
public:
    DINOv3ProbeHandler(const AppConfig& cfg) : config(cfg) {}

    GstPadProbeReturn handle_buffer(GstPad* pad, GstPadProbeInfo* info);

private:
    const AppConfig& config;

    // Helper functions
    bool should_process_batch(NvDsBatchMeta* batch_meta);
    NvDsInferTensorMeta* find_backbone_tensor(NvDsFrameMeta* frame_meta);
    int find_feature_layer_index(NvDsInferTensorMeta* tensor_meta);
    GstNvDsPreProcessBatchMeta* create_preprocess_meta(
        NvDsInferTensorMeta* tensor_meta,
        NvDsFrameMeta* frame_meta,
        int feat_idx);
};

// C-style callback wrapper for GStreamer
GstPadProbeReturn dinov3_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data);
