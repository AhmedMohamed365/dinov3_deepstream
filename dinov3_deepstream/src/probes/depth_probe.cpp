#include "depth_probe.h"
#include "utils/gst_utils.h"
#include <iostream>

bool DepthProbeHandler::should_debug() const {
    return config.debug.enabled &&
           ((call_idx <= config.debug.initial_frames) ||
            (call_idx % config.debug.periodic_interval == 0));
}

bool DepthProbeHandler::ensure_buffer_writable(GstPadProbeInfo* info) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return false;

    if (!gst_buffer_is_writable(buf)) {
        GstBuffer* wbuf = gst_buffer_make_writable(buf);
        if (!wbuf) return false;
        GST_PAD_PROBE_INFO_DATA(info) = wbuf;
    }
    return true;
}

NvBufSurface* DepthProbeHandler::map_buffer(GstBuffer* buf, GstMapInfo& map_info) {
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        return nullptr;
    }
    return (NvBufSurface*)map_info.data;
}

bool DepthProbeHandler::validate_surface(NvBufSurface* surface, bool debug) {
    if (!surface) return false;

    if (debug) {
        std::cout << "\n[DEPTH/CUDA] call #" << call_idx
                  << " memType=" << (int)surface->memType
                  << " numFilled=" << surface->numFilled
                  << " batchSize=" << surface->batchSize
                  << "\n";
    }

    return surface->memType == NVBUF_MEM_CUDA_DEVICE;
}

bool DepthProbeHandler::extract_depth_tensor(
    NvDsBatchMeta* batch_meta,
    NvDsFrameMeta* frame_meta,
    GstNvDsPreProcessBatchMeta* pbm,
    DepthTensorInfo& out_info,
    bool debug)
{
    // Find depth tensor meta
    NvDsInferTensorMeta* depth_tmeta = nullptr;
    for (size_t r = 0; r < pbm->roi_vector.size(); ++r) {
        auto& roi_meta = pbm->roi_vector[r];
        depth_tmeta = find_tensor_meta_in_user_meta_list(
            roi_meta.roi_user_meta_list,
            config.inference_ids.depth_uid);
        if (depth_tmeta) break;
    }

    if (!depth_tmeta) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] frame=" << frame_meta->frame_num
                      << " no depth tensor meta\n";
        }
        return false;
    }

    // Find depth layer
    int li = find_layer_index(depth_tmeta, config.layer_names.depth);
    if (li < 0) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] frame=" << frame_meta->frame_num
                      << " depth layer '" << config.layer_names.depth
                      << "' not found\n";
        }
        return false;
    }

    NvDsInferLayerInfo& layer = depth_tmeta->output_layers_info[li];

    // Validate dtype
    if (layer.dataType != NvDsInferDataType::FLOAT) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] frame=" << frame_meta->frame_num
                      << " depth dtype not FLOAT\n";
        }
        return false;
    }

    // Get device pointer
    void* depth_dev_void = (depth_tmeta->out_buf_ptrs_dev ?
                           depth_tmeta->out_buf_ptrs_dev[li] : nullptr);
    if (!depth_dev_void) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] frame=" << frame_meta->frame_num
                      << " no out_buf_ptrs_dev\n";
        }
        return false;
    }

    // Extract dimensions
    int H = 0, W = 0;
    if (layer.inferDims.numDims >= 2) {
        H = layer.inferDims.d[layer.inferDims.numDims - 2];
        W = layer.inferDims.d[layer.inferDims.numDims - 1];
    }

    if (H <= 0 || W <= 0) return false;

    out_info.depth_dev = (const float*)depth_dev_void;
    out_info.width = W;
    out_info.height = H;
    return true;
}

bool DepthProbeHandler::process_frame_depth(
    NvBufSurface* surface,
    NvDsFrameMeta* frame_meta,
    const DepthTensorInfo& depth_info,
    bool debug)
{
    const int b = (int)frame_meta->batch_id;
    if (b < 0 || b >= (int)surface->batchSize) return false;

    NvBufSurfaceParams& sl = surface->surfaceList[b];

    // Validate color format
    if (sl.colorFormat != NVBUF_COLOR_FORMAT_NV12 &&
        sl.colorFormat != NVBUF_COLOR_FORMAT_NV12_ER) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] frame=" << frame_meta->frame_num
                      << " surface not NV12 (fmt=" << (int)sl.colorFormat << ")\n";
        }
        return false;
    }

    // Get NV12 surface pointers
    uint8_t* base = (uint8_t*)sl.dataPtr;
    int pitchY = (int)sl.pitch;
    int pitchUV = pitchY;
    int outW = (int)sl.width;
    int outH = (int)sl.height;

    uint8_t* y_dev = base;
    uint8_t* uv_dev = base + (size_t)pitchY * (size_t)outH;

    // Launch CUDA kernel
    cudaStream_t stream = 0;
    cudaError_t e = depth_to_nv12_colormap_launch(
        depth_info.depth_dev, depth_info.width, depth_info.height,
        y_dev, uv_dev,
        outW, outH, pitchY, pitchUV,
        config.depth_range.near_m, config.depth_range.far_m,
        stream);

    if (e != cudaSuccess) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] launch error: "
                      << cudaGetErrorString(e) << "\n";
        }
        return false;
    }

    cudaError_t e2 = cudaStreamSynchronize(stream);
    if (e2 != cudaSuccess) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] sync error: "
                      << cudaGetErrorString(e2) << "\n";
        }
        return false;
    }

    if (debug) {
        std::cout << "[DEPTH/CUDA] frame=" << frame_meta->frame_num
                  << " wrote NV12 from depth " << depth_info.width << "x" << depth_info.height
                  << " -> " << outW << "x" << outH
                  << " pitch=" << pitchY
                  << " near=" << config.depth_range.near_m
                  << " far=" << config.depth_range.far_m
                  << "\n";
    }

    return true;
}

GstPadProbeReturn DepthProbeHandler::handle_buffer(
    GstPad* pad,
    GstPadProbeInfo* info)
{
    (void)pad;

    call_idx++;
    const bool debug = should_debug();

    if (!ensure_buffer_writable(info)) {
        return GST_PAD_PROBE_OK;
    }

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    GstMapInfo in_map{};
    NvBufSurface* surface = map_buffer(buf, in_map);
    if (!surface) return GST_PAD_PROBE_OK;

    if (!validate_surface(surface, debug)) {
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_OK;
    }

    // CRITICAL: Copy surface to ensure independence from other branches
    // The tee element shares the same NvBufSurface across branches, so we must copy
    NvBufSurface* new_surface = nullptr;
    NvBufSurfaceCreateParams create_params{};
    create_params.gpuId = surface->gpuId;
    create_params.width = surface->surfaceList[0].width;
    create_params.height = surface->surfaceList[0].height;
    create_params.size = 0;  // Auto-calculate
    create_params.colorFormat = surface->surfaceList[0].colorFormat;
    create_params.layout = surface->surfaceList[0].layout;
    create_params.memType = surface->memType;

    if (NvBufSurfaceCreate(&new_surface, surface->batchSize, &create_params) != 0) {
        if (debug) std::cerr << "[DEPTH] Failed to create surface copy\n";
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_OK;
    }

    // Copy surface contents
    if (NvBufSurfaceCopy(surface, new_surface) != 0) {
        if (debug) std::cerr << "[DEPTH] Failed to copy surface\n";
        NvBufSurfaceDestroy(new_surface);
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_OK;
    }

    // Unmap old surface and update buffer to point to new surface
    gst_buffer_unmap(buf, &in_map);

    // Remove old memory from buffer
    gst_buffer_remove_all_memory(buf);

    // Wrap new surface in GstMemory and add to buffer
    GstMemory* mem = gst_memory_new_wrapped(
        (GstMemoryFlags)(GST_MEMORY_FLAG_READONLY | GST_MEMORY_FLAG_NO_SHARE),
        new_surface,
        sizeof(NvBufSurface),
        0,
        sizeof(NvBufSurface),
        new_surface,
        [](gpointer data) { NvBufSurfaceDestroy((NvBufSurface*)data); });

    gst_buffer_append_memory(buf, mem);

    // Re-map the new surface
    if (!gst_buffer_map(buf, &in_map, GST_MAP_READ)) {
        return GST_PAD_PROBE_OK;
    }
    surface = (NvBufSurface*)in_map.data;

    auto* pbm = find_preprocess_meta_for_uid(batch_meta, config.inference_ids.depth_uid);
    if (!pbm) {
        if (debug) {
            std::cout << "[DEPTH/CUDA] no NVDS_PREPROCESS_BATCH_META for depth uid\n";
        }
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_OK;
    }

    int updated = 0;
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
        if (!fmeta) continue;

        DepthTensorInfo depth_info;
        if (!extract_depth_tensor(batch_meta, fmeta, pbm, depth_info, debug)) {
            continue;
        }

        if (process_frame_depth(surface, fmeta, depth_info, debug)) {
            updated++;
        }
    }

    if (debug && updated == 0) {
        std::cout << "[DEPTH/CUDA] no frames updated\n";
    }

    gst_buffer_unmap(buf, &in_map);
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn depth_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* handler = reinterpret_cast<DepthProbeHandler*>(user_data);
    return handler->handle_buffer(pad, info);
}
