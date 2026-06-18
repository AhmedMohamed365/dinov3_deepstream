#include "dinov3_probe.h"
#include "utils/gst_utils.h"
#include "utils/fps_tracker.h"
#include <cstring>

bool DINOv3ProbeHandler::should_process_batch(NvDsBatchMeta* batch_meta) {
    return !already_has_preprocess_for_uids(
        batch_meta,
        {config.inference_ids.depth_uid,
         config.inference_ids.detection_uid,
         config.inference_ids.segmentation_uid}
    );
}

NvDsInferTensorMeta* DINOv3ProbeHandler::find_backbone_tensor(
    NvDsFrameMeta* frame_meta)
{
    for (NvDsMetaList* l_user = frame_meta->frame_user_meta_list;
         l_user != nullptr;
         l_user = l_user->next)
    {
        NvDsUserMeta* user_meta = (NvDsUserMeta*)l_user->data;
        if (!user_meta) continue;

        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
            return (NvDsInferTensorMeta*)user_meta->user_meta_data;
        }
    }
    return nullptr;
}

int DINOv3ProbeHandler::find_feature_layer_index(
    NvDsInferTensorMeta* tensor_meta)
{
    for (unsigned i = 0; i < tensor_meta->num_output_layers; ++i) {
        const char* lname = tensor_meta->output_layers_info[i].layerName;
        if (lname && config.layer_names.features == lname) {
            return (int)i;
        }
    }
    return 0; // Default to first layer
}

GstNvDsPreProcessBatchMeta* DINOv3ProbeHandler::create_preprocess_meta(
    NvDsInferTensorMeta* tensor_meta,
    NvDsFrameMeta* frame_meta,
    int feat_idx)
{
    NvDsInferLayerInfo& layer = tensor_meta->output_layers_info[feat_idx];
    void* dev_ptr = tensor_meta->out_buf_ptrs_dev[feat_idx];

    const size_t elem_sz = elem_size_from_infer_dtype(layer.dataType);
    const size_t vol = volume_from_dims(layer.inferDims);
    const size_t bytes = elem_sz ? (vol * elem_sz) : 0;

    auto* pbm = new GstNvDsPreProcessBatchMeta();
    pbm->private_data = nullptr;
    pbm->target_unique_ids = {
        config.inference_ids.depth_uid,
        config.inference_ids.detection_uid,
        config.inference_ids.segmentation_uid,
        config.inference_ids.optical_flow_uid
    };

    // ROI info (full-frame)
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
    pbm->roi_vector.clear();
    pbm->roi_vector.push_back(roi_meta);

    // Tensor meta
    pbm->tensor_meta = new NvDsPreProcessTensorMeta();
    pbm->tensor_meta->raw_tensor_buffer = dev_ptr;
    pbm->tensor_meta->buffer_size = bytes;
    pbm->tensor_meta->gpu_id = tensor_meta->gpu_id;
    pbm->tensor_meta->data_type = (layer.dataType == NvDsInferDataType::HALF)
        ? NvDsDataType_FP16
        : NvDsDataType_FP32;
    pbm->tensor_meta->tensor_name = config.layer_names.features;
    pbm->tensor_meta->private_data = nullptr;
    pbm->tensor_meta->meta_id = 0;
    pbm->tensor_meta->maintain_aspect_ratio = FALSE;

    // tensor_shape: include batch dim, then layer dims
    pbm->tensor_meta->tensor_shape.clear();
    pbm->tensor_meta->tensor_shape.push_back(1);
    for (int d = 0; d < layer.inferDims.numDims; ++d) {
        pbm->tensor_meta->tensor_shape.push_back(layer.inferDims.d[d]);
    }

    return pbm;
}

GstPadProbeReturn DINOv3ProbeHandler::handle_buffer(
    GstPad* pad,
    GstPadProbeInfo* info)
{
    (void)pad;

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    static int frame_count = 0;
    frame_count++;
    const bool should_log = (frame_count % 30 == 0);
    if (should_log) {
        std::cerr << "[DEBUG DINOv3] Probe handle_buffer called. frame_count=" << frame_count << std::endl;
        GstMapInfo in_map{};
        if (gst_buffer_map(buf, &in_map, GST_MAP_READ)) {
            NvBufSurface* surface = (NvBufSurface*)in_map.data;
            if (surface && surface->numFilled > 0) {
                NvBufSurfaceParams& sl = surface->surfaceList[0];
                uint8_t host_pixels[10];
                cudaError_t err = cudaMemcpy(host_pixels, sl.dataPtr, 10 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                if (err == cudaSuccess) {
                    std::cerr << "[DEBUG DINOv3] Input Y pixels: ";
                    for (int i=0; i<10; ++i) std::cerr << (int)host_pixels[i] << " ";
                    std::cerr << std::endl;
                } else {
                    std::cerr << "[DEBUG DINOv3] Input pixels cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
                }
            }
            gst_buffer_unmap(buf, &in_map);
        }
    }

    nvds_acquire_meta_lock(batch_meta);

    if (!should_process_batch(batch_meta)) {
        if (should_log) {
            std::cerr << "[DEBUG DINOv3] batch already processed, skipping" << std::endl;
        }
        nvds_release_meta_lock(batch_meta);
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        if (!frame_meta) continue;

        NvDsInferTensorMeta* tensor_meta = find_backbone_tensor(frame_meta);
        if (!tensor_meta) {
            if (should_log) {
                std::cerr << "[DEBUG DINOv3] NvDsInferTensorMeta NOT found in frame_meta!" << std::endl;
            }
            continue;
        }
        if (!tensor_meta->num_output_layers) {
            if (should_log) {
                std::cerr << "[DEBUG DINOv3] tensor_meta has 0 output layers!" << std::endl;
            }
            continue;
        }
        if (!tensor_meta->out_buf_ptrs_dev) {
            if (should_log) {
                std::cerr << "[DEBUG DINOv3] tensor_meta out_buf_ptrs_dev is null!" << std::endl;
            }
            continue;
        }

        int feat_idx = find_feature_layer_index(tensor_meta);
        if (!tensor_meta->out_buf_ptrs_dev[feat_idx]) {
            if (should_log) {
                std::cerr << "[DEBUG DINOv3] feature layer buffer pointer is null!" << std::endl;
            }
            continue;
        }

        if (should_log) {
            std::cerr << "[DEBUG DINOv3] Successfully found backbone feature tensor. DataType=" << (int)tensor_meta->output_layers_info[feat_idx].dataType << std::endl;
            void* dev_ptr = tensor_meta->out_buf_ptrs_dev[feat_idx];
            int vol = volume_from_dims(tensor_meta->output_layers_info[feat_idx].inferDims);
            if (tensor_meta->output_layers_info[feat_idx].dataType == NvDsInferDataType::FLOAT) {
                std::vector<float> host_features(vol);
                cudaMemcpy(host_features.data(), dev_ptr, vol * sizeof(float), cudaMemcpyDeviceToHost);
                float min_val = host_features[0];
                float max_val = host_features[0];
                double sum_val = 0;
                for (int i=0; i<vol; ++i) {
                    if (host_features[i] < min_val) min_val = host_features[i];
                    if (host_features[i] > max_val) max_val = host_features[i];
                    sum_val += host_features[i];
                }
                std::cerr << "[DEBUG DINOv3] Features stats (frame=" << frame_meta->frame_num << ", ptr=" << dev_ptr << "): vol=" << vol
                          << " min=" << min_val
                          << " max=" << max_val
                          << " mean=" << (sum_val / vol)
                          << " first_5=[" << host_features[0] << ", " << host_features[1] << ", " << host_features[2] << ", " << host_features[3] << ", " << host_features[4] << "]"
                          << std::endl;
            } else {
                std::cerr << "[DEBUG DINOv3] Backbone features dataType is not FLOAT!" << std::endl;
            }
        }

        GstNvDsPreProcessBatchMeta* pbm = create_preprocess_meta(
            tensor_meta, frame_meta, feat_idx);

        NvDsUserMeta* bm_umeta = nvds_acquire_user_meta_from_pool(batch_meta);
        if (!bm_umeta) {
            delete pbm->tensor_meta;
            delete pbm;
            continue;
        }

        bm_umeta->user_meta_data = (void*)pbm;
        bm_umeta->base_meta.meta_type = (NvDsMetaType)NVDS_PREPROCESS_BATCH_META;
        bm_umeta->base_meta.copy_func = preprocess_batchmeta_copy_func;
        bm_umeta->base_meta.release_func = preprocess_batchmeta_release_func;
        bm_umeta->base_meta.batch_meta = batch_meta;

        nvds_add_user_meta_to_batch(batch_meta, bm_umeta);
        break;
    }

    FPSTracker::getInstance().update("Backbone", batch_meta->num_frames_in_batch);
    nvds_release_meta_lock(batch_meta);
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn dinov3_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* handler = reinterpret_cast<DINOv3ProbeHandler*>(user_data);
    return handler->handle_buffer(pad, info);
}
