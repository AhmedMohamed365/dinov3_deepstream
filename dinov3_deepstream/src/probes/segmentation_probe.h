#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"
#include "utils_cuda/segmentation.h"
#include <vector>
#include <string>

// Context for segmentation visualization
struct SegmentationContext {
    std::vector<std::string> class_names;
    AppConfig config;
};

// Segmentation probe handler
// Visualizes semantic segmentation with colorization and text labels
class SegmentationProbeHandler {
public:
    SegmentationProbeHandler(SegmentationContext* ctx)
        : context(ctx) {}

    ~SegmentationProbeHandler();

    GstPadProbeReturn handle_buffer(GstPad* pad, GstPadProbeInfo* info);

    // Public access to context for cleanup
    SegmentationContext* context;

private:

    // Persistent GPU buffers
    struct GpuBuffers {
        int32_t* class_map_dev = nullptr;
        size_t class_map_bytes = 0;

        int32_t* count_dev = nullptr;
        int64_t* sumx_dev = nullptr;
        int64_t* sumy_dev = nullptr;
        size_t stats_bytes = 0;

        void ensure_class_map(size_t H, size_t W);
        void ensure_stats_buffers(size_t num_classes);
        void cleanup();
    } gpu_buffers;

    struct SegmentationTensorInfo {
        void* logits_dev;
        bool is_half;
        int C, H, W;
    };

    bool ensure_buffer_writable(GstPadProbeInfo* info);
    NvBufSurface* map_buffer(GstBuffer* buf, GstMapInfo& map_info);

    bool extract_segmentation_tensor(
        NvDsBatchMeta* batch_meta,
        NvDsFrameMeta* frame_meta,
        GstNvDsPreProcessBatchMeta* pbm,
        SegmentationTensorInfo& out_info);

    bool colorize_frame(
        NvBufSurface* surface,
        int batch_id,
        const SegmentationTensorInfo& seg_info,
        cudaStream_t stream);

    void add_label_overlays(
        NvDsBatchMeta* batch_meta,
        NvDsFrameMeta* frame_meta,
        const SegmentationTensorInfo& seg_info,
        int outW, int outH,
        cudaStream_t stream);
};

// C-style callback wrapper for GStreamer
GstPadProbeReturn seg_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data);

// Context destructor
void segmentation_context_destroy(gpointer data);
