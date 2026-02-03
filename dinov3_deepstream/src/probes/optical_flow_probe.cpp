#include "optical_flow_probe.h"
#include "utils/gst_utils.h"
#include "utils_cuda/optical_flow.h"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

// ============================================================================
// OpticalFlowPreprocessHandler - SINK pad probe
// Concatenates two consecutive DINOv3 features before optical flow inference
// ============================================================================

OpticalFlowPreprocessHandler::~OpticalFlowPreprocessHandler() {
    if (prev_features_dev) {
        cudaFree(prev_features_dev);
        prev_features_dev = nullptr;
    }
    if (concat_features_dev) {
        cudaFree(concat_features_dev);
        concat_features_dev = nullptr;
    }
}

bool OpticalFlowPreprocessHandler::should_debug() const {
    if (!config.debug.enabled) return false;
    return frame_count < config.debug.initial_frames ||
           (frame_count % config.debug.periodic_interval == 0);
}

bool OpticalFlowPreprocessHandler::extract_dinov3_features(
    NvDsBatchMeta* batch_meta,
    FeatureTensorInfo& out_info,
    bool debug)
{
    // Find the SHARED preprocessing metadata (created by dinov3_probe)
    // Use depth_uid to find it since optical_flow_uid gets removed after first use
    auto* pbm = find_preprocess_meta_for_uid(batch_meta, config.inference_ids.depth_uid);
    if (!pbm) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] No shared preprocessing meta found\n";
        return false;
    }

    // Extract the features tensor pointer from the preprocessing metadata
    if (!pbm->tensor_meta || !pbm->tensor_meta->raw_tensor_buffer) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] No tensor in preprocessing meta\n";
        return false;
    }

    // Get tensor dimensions from preprocessing metadata
    if (pbm->tensor_meta->tensor_shape.size() < 3) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] Invalid tensor shape\n";
        return false;
    }

    // Assuming shape is [1, C, H, W] or [C, H, W]
    size_t num_dims = pbm->tensor_meta->tensor_shape.size();
    int C = pbm->tensor_meta->tensor_shape[num_dims - 3];
    int H = pbm->tensor_meta->tensor_shape[num_dims - 2];
    int W = pbm->tensor_meta->tensor_shape[num_dims - 1];

    if (H <= 0 || W <= 0 || C <= 0) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] Invalid dimensions: "
                            << C << "x" << H << "x" << W << "\n";
        return false;
    }

    out_info.features_dev = pbm->tensor_meta->raw_tensor_buffer;
    out_info.H = H;
    out_info.W = W;
    out_info.C = C;

    if (debug) {
        std::cout << "[OPTICAL_FLOW_PREPROCESS] Extracted features from preprocessing meta: "
                  << C << "x" << H << "x" << W << "\n";
    }

    return true;
}

void OpticalFlowPreprocessHandler::buffer_features(const FeatureTensorInfo& features) {
    size_t elem_size = sizeof(float);  // Assuming FP32 features
    size_t needed_size = (size_t)features.C * features.H * features.W * elem_size;

    // Reallocate if needed
    if (!prev_features_dev || prev_features_size < needed_size) {
        if (prev_features_dev) cudaFree(prev_features_dev);
        cudaMalloc(&prev_features_dev, needed_size);
        prev_features_size = needed_size;
    }

    // Copy current features to buffer (device-to-device)
    cudaMemcpy(prev_features_dev, features.features_dev, needed_size, cudaMemcpyDeviceToDevice);
    prev_H = features.H;
    prev_W = features.W;
    prev_C = features.C;
}

bool OpticalFlowPreprocessHandler::concatenate_and_update_meta(
    NvDsBatchMeta* batch_meta,
    const FeatureTensorInfo& current_features,
    bool debug)
{
    // Allocate concatenated buffer [2*C, H, W]
    int H = current_features.H;
    int W = current_features.W;
    int C = current_features.C;
    size_t single_feature_size = (size_t)C * H * W * sizeof(float);
    size_t concat_size = 2 * single_feature_size;

    if (!concat_features_dev || concat_features_size < concat_size) {
        if (concat_features_dev) cudaFree(concat_features_dev);
        cudaMalloc(&concat_features_dev, concat_size);
        concat_features_size = concat_size;
    }

    // Concatenate: [prev_features; current_features] along channel dimension
    // Result layout: [C channels from prev, C channels from current, H, W]
    float* concat_ptr = (float*)concat_features_dev;

    cudaMemcpy(concat_ptr, prev_features_dev, single_feature_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(concat_ptr + (C * H * W), current_features.features_dev, single_feature_size, cudaMemcpyDeviceToDevice);

    if (debug) {
        std::cout << "[OPTICAL_FLOW_PREPROCESS] Concatenated features: "
                  << "prev[" << C << "," << H << "," << W << "] + "
                  << "curr[" << C << "," << H << "," << W << "] -> "
                  << "concat[" << (2*C) << "," << H << "," << W << "]\n";
    }

    // Find the shared preprocessing meta (to copy settings from)
    auto* shared_pbm = find_preprocess_meta_for_uid(batch_meta, config.inference_ids.optical_flow_uid);
    if (!shared_pbm) {
        if (debug) {
            std::cout << "[OPTICAL_FLOW_PREPROCESS] No shared preprocessing meta found\n";
        }
        return false;
    }

    // Get frame_meta for ROI initialization
    NvDsFrameMeta* frame_meta = nullptr;
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta*)l_frame->data;
        if (frame_meta) break;
    }
    if (!frame_meta) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] No frame meta found\n";
        return false;
    }

    // Remove optical_flow_uid from shared metadata to prevent it from using wrong tensor
    auto& targets = shared_pbm->target_unique_ids;
    targets.erase(
        std::remove(targets.begin(), targets.end(), config.inference_ids.optical_flow_uid),
        targets.end()
    );

    if (debug) {
        std::cout << "[OPTICAL_FLOW_PREPROCESS] Removed optical_flow_uid from shared metadata\n";
    }

    // Create SEPARATE preprocessing metadata for optical flow
    auto* new_pbm = new GstNvDsPreProcessBatchMeta();
    new_pbm->private_data = nullptr;
    new_pbm->target_unique_ids = {config.inference_ids.optical_flow_uid};

    // Create ROI metadata (full-frame)
    NvDsRoiMeta roi_meta;
    std::memset(&roi_meta, 0, sizeof(roi_meta));
    roi_meta.roi.left = 0;
    roi_meta.roi.top = 0;
    roi_meta.roi.width = frame_meta->pipeline_width;
    roi_meta.roi.height = frame_meta->pipeline_height;
    roi_meta.scale_ratio_x = 1.0f;
    roi_meta.scale_ratio_y = 1.0f;
    roi_meta.offset_left = 0;
    roi_meta.offset_top = 0;
    roi_meta.frame_meta = frame_meta;
    new_pbm->roi_vector.clear();
    new_pbm->roi_vector.push_back(roi_meta);

    // Create tensor meta with concatenated features
    new_pbm->tensor_meta = new NvDsPreProcessTensorMeta();
    new_pbm->tensor_meta->raw_tensor_buffer = concat_features_dev;
    new_pbm->tensor_meta->buffer_size = concat_features_size;
    new_pbm->tensor_meta->gpu_id = shared_pbm->tensor_meta->gpu_id;
    new_pbm->tensor_meta->data_type = shared_pbm->tensor_meta->data_type;
    new_pbm->tensor_meta->tensor_name = config.layer_names.features;
    new_pbm->tensor_meta->private_data = nullptr;
    new_pbm->tensor_meta->meta_id = 0;
    new_pbm->tensor_meta->maintain_aspect_ratio = FALSE;

    // Set tensor shape: [1, 2*C, H, W]
    new_pbm->tensor_meta->tensor_shape.clear();
    new_pbm->tensor_meta->tensor_shape.push_back(1);
    new_pbm->tensor_meta->tensor_shape.push_back(2 * C);
    new_pbm->tensor_meta->tensor_shape.push_back(H);
    new_pbm->tensor_meta->tensor_shape.push_back(W);

    if (debug) {
        std::cout << "[OPTICAL_FLOW_PREPROCESS] Created separate preprocessing meta: "
                  << "shape=[1," << (2*C) << "," << H << "," << W << "], "
                  << "buffer_size=" << concat_features_size << " bytes\n";
    }

    // Add separate preprocessing metadata to batch
    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) {
        delete new_pbm->tensor_meta;
        delete new_pbm;
        return false;
    }

    user_meta->user_meta_data = (void*)new_pbm;
    user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_PREPROCESS_BATCH_META;
    user_meta->base_meta.copy_func = preprocess_batchmeta_copy_func;
    user_meta->base_meta.release_func = preprocess_batchmeta_release_func;
    user_meta->base_meta.batch_meta = batch_meta;

    nvds_add_user_meta_to_batch(batch_meta, user_meta);

    return true;
}

GstPadProbeReturn OpticalFlowPreprocessHandler::handle_buffer(
    GstPad* pad,
    GstPadProbeInfo* info)
{
    (void)pad;

    frame_count++;
    const bool debug = should_debug();

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    // Extract current DINOv3 features
    FeatureTensorInfo current_features;
    if (!extract_dinov3_features(batch_meta, current_features, debug)) {
        return GST_PAD_PROBE_OK;
    }

    // First frame: just buffer features and drop buffer (skip inference)
    if (frame_count == 1 || !prev_features_dev) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] Frame 1: buffering features, dropping buffer\n";
        buffer_features(current_features);
        return GST_PAD_PROBE_DROP;  // Drop buffer to prevent inference with non-concatenated features
    }

    // Check dimensions match
    if (prev_H != current_features.H || prev_W != current_features.W || prev_C != current_features.C) {
        if (debug) std::cout << "[OPTICAL_FLOW_PREPROCESS] Dimension mismatch, re-buffering and dropping\n";
        buffer_features(current_features);
        return GST_PAD_PROBE_DROP;  // Drop buffer to prevent inference with mismatched features
    }

    // Concatenate previous and current features, update preprocessing meta
    concatenate_and_update_meta(batch_meta, current_features, debug);

    // Buffer current features for next iteration
    buffer_features(current_features);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn optical_flow_sink_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* handler = reinterpret_cast<OpticalFlowPreprocessHandler*>(user_data);
    return handler->handle_buffer(pad, info);
}

// ============================================================================
// OpticalFlowVisualizationHandler - SRC pad probe
// Visualizes optical flow output after inference
// ============================================================================

bool OpticalFlowVisualizationHandler::should_debug() const {
    if (!config.debug.enabled) return false;
    return frame_count < config.debug.initial_frames ||
           (frame_count % config.debug.periodic_interval == 0);
}

bool OpticalFlowVisualizationHandler::ensure_buffer_writable(GstPadProbeInfo* info) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return false;

    if (!gst_buffer_is_writable(buf)) {
        GstBuffer* wbuf = gst_buffer_make_writable(buf);
        if (!wbuf) return false;
        GST_PAD_PROBE_INFO_DATA(info) = wbuf;
    }
    return true;
}

NvBufSurface* OpticalFlowVisualizationHandler::map_buffer(GstBuffer* buf, GstMapInfo& map_info) {
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        return nullptr;
    }
    return (NvBufSurface*)map_info.data;
}

bool OpticalFlowVisualizationHandler::validate_surface(NvBufSurface* surface, bool debug) {
    if (!surface || surface->batchSize == 0) {
        if (debug) std::cout << "[OPTICAL_FLOW_VIZ] Invalid surface\n";
        return false;
    }
    if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
        if (debug) std::cout << "[OPTICAL_FLOW_VIZ] Surface not on GPU\n";
        return false;
    }
    return true;
}

bool OpticalFlowVisualizationHandler::extract_optical_flow_tensor(
    NvDsBatchMeta* batch_meta,
    NvDsFrameMeta* frame_meta,
    GstNvDsPreProcessBatchMeta* pbm,
    OpticalFlowTensorInfo& out_info,
    bool debug)
{
    (void)frame_meta;  // Not used currently

    if (!pbm) {
        if (debug) std::cout << "[OPTICAL_FLOW_VIZ] No preprocessing metadata provided\n";
        return false;
    }

    // Find optical flow tensor meta in ROI user meta list
    // nvinfer attaches outputs to ROI's user_meta_list when using input-tensor-from-meta
    if (debug) {
        std::cout << "[OPTICAL_FLOW_VIZ] ROI vector size: " << pbm->roi_vector.size() << "\n";
        for (size_t r = 0; r < pbm->roi_vector.size(); ++r) {
            auto& roi_meta = pbm->roi_vector[r];
            std::cout << "  ROI " << r << " user_meta_list:\n";
            for (NvDsMetaList* l = roi_meta.roi_user_meta_list; l; l = l->next) {
                NvDsUserMeta* um = (NvDsUserMeta*)l->data;
                if (um && um->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                    auto* tm = (NvDsInferTensorMeta*)um->user_meta_data;
                    if (tm) {
                        std::cout << "    Found tensor output with UID: " << tm->unique_id << "\n";
                    }
                }
            }
        }
    }

    NvDsInferTensorMeta* tmeta = nullptr;
    for (size_t r = 0; r < pbm->roi_vector.size(); ++r) {
        auto& roi_meta = pbm->roi_vector[r];
        tmeta = find_tensor_meta_in_user_meta_list(
            roi_meta.roi_user_meta_list,
            config.inference_ids.optical_flow_uid);
        if (tmeta) break;
    }

    if (!tmeta) {
        if (debug) std::cout << "[OPTICAL_FLOW_VIZ] No optical flow tensor meta (UID "
                            << config.inference_ids.optical_flow_uid << ") in ROI vector\n";
        return false;
    }

    // Find optical flow layer
    int li = find_layer_index(tmeta, config.layer_names.optical_flow);
    if (li < 0) {
        if (debug) std::cout << "[OPTICAL_FLOW_VIZ] Optical flow layer not found\n";
        return false;
    }

    NvDsInferLayerInfo& layer = tmeta->output_layers_info[li];
    void* flow_dev_void = (tmeta->out_buf_ptrs_dev ? tmeta->out_buf_ptrs_dev[li] : nullptr);
    if (!flow_dev_void) {
        if (debug) std::cout << "[OPTICAL_FLOW_VIZ] Flow tensor pointer is null\n";
        return false;
    }

    // Extract dimensions [2, H, W]
    int H = 0, W = 0;
    if (layer.inferDims.numDims >= 2) {
        H = layer.inferDims.d[layer.inferDims.numDims - 2];
        W = layer.inferDims.d[layer.inferDims.numDims - 1];
    }

    if (debug) {
        std::cout << "[OPTICAL_FLOW_VIZ] Found optical flow tensor (UID " << tmeta->unique_id << "): ";
        std::cout << "numDims=" << layer.inferDims.numDims << ", shape=[";
        for (int d = 0; d < layer.inferDims.numDims; ++d) {
            std::cout << layer.inferDims.d[d];
            if (d < layer.inferDims.numDims - 1) std::cout << ",";
        }
        std::cout << "], extracted H=" << H << ", W=" << W << "\n";
    }

    if (H <= 0 || W <= 0) return false;

    out_info.flow_dev = (const float*)flow_dev_void;
    out_info.width = W;
    out_info.height = H;
    return true;
}

bool OpticalFlowVisualizationHandler::process_frame_optical_flow(
    NvBufSurface* surface,
    NvDsFrameMeta* frame_meta,
    const OpticalFlowTensorInfo& flow_info,
    bool debug)
{
    const int b = (int)frame_meta->batch_id;
    if (b < 0 || b >= (int)surface->batchSize) return false;

    NvBufSurfaceParams& sl = surface->surfaceList[b];

    // Validate color format - must be NV12 family
    if (!is_nv12_color_format(sl.colorFormat)) {
        if (debug) {
            std::cout << "[OPTICAL_FLOW_VIZ/CUDA] frame=" << frame_meta->frame_num
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

    // Launch CUDA kernel (color visualization with Middlebury color wheel)
    cudaStream_t stream = 0;
    float max_flow = 20.0f;  // Maximum expected flow magnitude in pixels

    cudaError_t e = optical_flow_to_nv12_color_launch(
        flow_info.flow_dev, flow_info.width, flow_info.height,
        y_dev, uv_dev,
        outW, outH, pitchY, pitchUV,
        max_flow,
        stream);
    if (e != cudaSuccess) {
        if (debug) {
            std::cout << "[OPTICAL_FLOW_VIZ/CUDA] launch error: "
                      << cudaGetErrorString(e) << "\n";
        }
        return false;
    }

    cudaError_t e2 = cudaStreamSynchronize(stream);
    if (e2 != cudaSuccess) {
        if (debug) {
            std::cout << "[OPTICAL_FLOW_VIZ/CUDA] sync error: "
                      << cudaGetErrorString(e2) << "\n";
        }
        return false;
    }

    if (debug) {
        std::cout << "[OPTICAL_FLOW_VIZ/CUDA] frame=" << frame_meta->frame_num
                  << " visualized flow " << flow_info.width << "x" << flow_info.height
                  << " -> " << outW << "x" << outH << "\n";
    }
    return true;
}

GstPadProbeReturn OpticalFlowVisualizationHandler::handle_buffer(
    GstPad* pad,
    GstPadProbeInfo* info)
{
    (void)pad;

    frame_count++;
    const bool debug = should_debug();

    if (!ensure_buffer_writable(info)) {
        return GST_PAD_PROBE_DROP;
    }

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_DROP;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_DROP;

    GstMapInfo in_map{};
    NvBufSurface* surface = map_buffer(buf, in_map);
    if (!surface) return GST_PAD_PROBE_DROP;

    if (!validate_surface(surface, debug)) {
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_DROP;
    }

    // Copy surface to ensure independence from other branches
    surface = copy_and_replace_buffer_surface(buf, in_map, surface);
    if (!surface) {
        if (debug) std::cerr << "[OPTICAL_FLOW_VIZ] Failed to copy surface\n";
        return GST_PAD_PROBE_DROP;
    }

    // Debug: List all preprocessing metadata and their UIDs
    if (debug) {
        std::cout << "[OPTICAL_FLOW_VIZ] Searching for preprocessing metadata:\n";
        for (NvDsMetaList* l = batch_meta->batch_user_meta_list; l; l = l->next) {
            NvDsUserMeta* um = (NvDsUserMeta*)l->data;
            if (um && um->base_meta.meta_type == (NvDsMetaType)NVDS_PREPROCESS_BATCH_META) {
                auto* pbm_debug = (GstNvDsPreProcessBatchMeta*)um->user_meta_data;
                if (pbm_debug) {
                    std::cout << "  Found preprocess meta with UIDs: ";
                    for (auto uid : pbm_debug->target_unique_ids) {
                        std::cout << uid << " ";
                    }
                    std::cout << "\n";
                }
            }
        }
    }

    // Look for the SEPARATE optical flow preprocessing metadata (not the shared one)
    auto* pbm = find_preprocess_meta_for_uid(batch_meta, config.inference_ids.optical_flow_uid);
    if (!pbm) {
        if (debug) {
            std::cout << "[OPTICAL_FLOW_VIZ] No optical flow preprocessing meta found\n";
        }
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_DROP;
    }

    if (debug) {
        std::cout << "[OPTICAL_FLOW_VIZ] Using optical flow preprocessing meta (UIDs: ";
        for (auto uid : pbm->target_unique_ids) {
            std::cout << uid << " ";
        }
        std::cout << ")\n";
    }

    // Extract optical flow tensor and visualize
    int updated = 0;
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
        if (!fmeta) continue;

        OpticalFlowTensorInfo flow_info;
        if (!extract_optical_flow_tensor(batch_meta, fmeta, pbm, flow_info, debug)) {
            continue;
        }

        if (process_frame_optical_flow(surface, fmeta, flow_info, debug)) {
            updated++;
        }
    }

    if (debug && updated == 0) {
        std::cout << "[OPTICAL_FLOW_VIZ] No frames updated\n";
    }

    gst_buffer_unmap(buf, &in_map);
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn optical_flow_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* handler = reinterpret_cast<OpticalFlowVisualizationHandler*>(user_data);
    return handler->handle_buffer(pad, info);
}
