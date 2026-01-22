#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"
#include "utils_cuda/depth.h"

// Depth probe handler
// Visualizes depth maps by converting them to NV12 colormap
class DepthProbeHandler {
public:
    DepthProbeHandler(const AppConfig& cfg)
        : config(cfg), call_idx(0) {}

    GstPadProbeReturn handle_buffer(GstPad* pad, GstPadProbeInfo* info);

private:
    const AppConfig& config;
    uint64_t call_idx;

    // Helper functions
    bool should_debug() const;
    bool ensure_buffer_writable(GstPadProbeInfo* info);
    NvBufSurface* map_buffer(GstBuffer* buf, GstMapInfo& map_info);
    bool validate_surface(NvBufSurface* surface, bool debug);

    struct DepthTensorInfo {
        const float* depth_dev;
        int width;
        int height;
    };

    bool extract_depth_tensor(
        NvDsBatchMeta* batch_meta,
        NvDsFrameMeta* frame_meta,
        GstNvDsPreProcessBatchMeta* pbm,
        DepthTensorInfo& out_info,
        bool debug);

    bool process_frame_depth(
        NvBufSurface* surface,
        NvDsFrameMeta* frame_meta,
        const DepthTensorInfo& depth_info,
        bool debug);
};

// C-style callback wrapper for GStreamer
GstPadProbeReturn depth_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data);
