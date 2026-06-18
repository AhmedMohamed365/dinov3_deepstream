#include "segmentation_probe.h"
#include "utils/gst_utils.h"
#include "utils/fps_tracker.h"
#include "utils/segmentation_tracker.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include "nvds_opticalflow_meta.h"
#include <iostream>

// GpuBuffers methods
void SegmentationProbeHandler::GpuBuffers::ensure_class_map(size_t H, size_t W) {
    const size_t n_pix = H * W;
    const size_t need_bytes = n_pix * sizeof(int32_t);

    if (!class_map_dev || class_map_bytes < need_bytes) {
        if (class_map_dev) cudaFree(class_map_dev);
        cudaMalloc(&class_map_dev, need_bytes);
        class_map_bytes = need_bytes;
    }
    if (!last_class_map_dev || last_class_map_bytes < need_bytes) {
        if (last_class_map_dev) cudaFree(last_class_map_dev);
        cudaMalloc(&last_class_map_dev, need_bytes);
        last_class_map_bytes = need_bytes;
    }
}

void SegmentationProbeHandler::GpuBuffers::ensure_stats_buffers(size_t num_classes) {
    const size_t need_stats_bytes = num_classes * sizeof(int32_t);
    const size_t need_sum_bytes = num_classes * sizeof(int64_t);
    const size_t total_bytes = need_stats_bytes + 2 * need_sum_bytes;

    if (!count_dev || stats_bytes < total_bytes) {
        if (count_dev) cudaFree(count_dev);
        if (sumx_dev) cudaFree(sumx_dev);
        if (sumy_dev) cudaFree(sumy_dev);

        cudaMalloc(&count_dev, num_classes * sizeof(int32_t));
        cudaMalloc(&sumx_dev, num_classes * sizeof(int64_t));
        cudaMalloc(&sumy_dev, num_classes * sizeof(int64_t));
        stats_bytes = total_bytes;
    }
}

void SegmentationProbeHandler::GpuBuffers::cleanup() {
    if (class_map_dev) {
        cudaFree(class_map_dev);
        class_map_dev = nullptr;
    }
    if (last_class_map_dev) {
        cudaFree(last_class_map_dev);
        last_class_map_dev = nullptr;
    }
    if (count_dev) {
        cudaFree(count_dev);
        count_dev = nullptr;
    }
    if (sumx_dev) {
        cudaFree(sumx_dev);
        sumx_dev = nullptr;
    }
    if (sumy_dev) {
        cudaFree(sumy_dev);
        sumy_dev = nullptr;
    }
}

SegmentationProbeHandler::~SegmentationProbeHandler() {
    gpu_buffers.cleanup();
}

bool SegmentationProbeHandler::ensure_buffer_writable(GstPadProbeInfo* info) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return false;

    if (!gst_buffer_is_writable(buf)) {
        GstBuffer* wbuf = gst_buffer_make_writable(buf);
        if (!wbuf) return false;
        GST_PAD_PROBE_INFO_DATA(info) = wbuf;
    }
    return true;
}

NvBufSurface* SegmentationProbeHandler::map_buffer(GstBuffer* buf, GstMapInfo& map_info) {
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        return nullptr;
    }
    return (NvBufSurface*)map_info.data;
}

bool SegmentationProbeHandler::extract_segmentation_tensor(
    NvDsBatchMeta* batch_meta,
    NvDsFrameMeta* frame_meta,
    GstNvDsPreProcessBatchMeta* pbm,
    SegmentationTensorInfo& out_info)
{
    // Find segmentation tensor meta
    NvDsInferTensorMeta* tmeta = nullptr;
    for (size_t r = 0; r < pbm->roi_vector.size(); ++r) {
        auto& roi_meta = pbm->roi_vector[r];
        tmeta = find_tensor_meta_in_user_meta_list(
            roi_meta.roi_user_meta_list,
            context->config.inference_ids.segmentation_uid);
        if (tmeta) break;
    }
    if (!tmeta) return false;

    // Find segmentation layer
    int li = find_layer_index(tmeta, context->config.layer_names.semantic_segmentation);
    if (li < 0) return false;

    NvDsInferLayerInfo& layer = tmeta->output_layers_info[li];
    void* logits_dev = (tmeta->out_buf_ptrs_dev ? tmeta->out_buf_ptrs_dev[li] : nullptr);
    if (!logits_dev) return false;

    // Parse logits dimensions -> C,H,W
    int C = 0, H = 0, W = 0;
    if (layer.inferDims.numDims == 4) {
        C = layer.inferDims.d[1];
        H = layer.inferDims.d[2];
        W = layer.inferDims.d[3];
    } else if (layer.inferDims.numDims == 3) {
        C = layer.inferDims.d[0];
        H = layer.inferDims.d[1];
        W = layer.inferDims.d[2];
    } else {
        return false;
    }

    if (C <= 0 || H <= 0 || W <= 0) return false;

    // Validate data type
    const bool is_half = (layer.dataType == NvDsInferDataType::HALF);
    if (!(layer.dataType == NvDsInferDataType::FLOAT || layer.dataType == NvDsInferDataType::HALF)) {
        return false;
    }

    out_info.logits_dev = logits_dev;
    out_info.is_half = is_half;
    out_info.C = C;
    out_info.H = H;
    out_info.W = W;
    return true;
}

bool SegmentationProbeHandler::colorize_frame(
    NvBufSurface* surface,
    int batch_id,
    const SegmentationTensorInfo& seg_info,
    cudaStream_t stream)
{
    // Ensure GPU buffers are sized correctly
    gpu_buffers.ensure_class_map(seg_info.H, seg_info.W);
    gpu_buffers.ensure_stats_buffers(seg_info.C);

    // 1) Argmax: logits -> class_map_dev
    if (seg_argmax_launch(seg_info.logits_dev, seg_info.is_half,
                         seg_info.C, seg_info.H, seg_info.W,
                         gpu_buffers.class_map_dev, stream) != cudaSuccess) {
        return false;
    }

    // 2) Overwrite NV12 with segmentation colors
    NvBufSurfaceParams& sl = surface->surfaceList[batch_id];
    if (!is_nv12_color_format(sl.colorFormat)) {
        return false;
    }

    int outW = (int)sl.width;
    int outH = (int)sl.height;
    int pitch = (int)sl.pitch;

    uint8_t* base = (uint8_t*)sl.dataPtr;
    if (!base) return false;

    uint8_t* y_dev = base;
    uint8_t* uv_dev = base + (size_t)pitch * (size_t)outH;

    if (seg_classmap_to_nv12_launch(gpu_buffers.class_map_dev, seg_info.W, seg_info.H,
                                    y_dev, uv_dev,
                                    outW, outH, pitch, pitch,
                                    context->config.visualization.alpha,
                                    stream) != cudaSuccess) {
        return false;
    }

    return true;
}

void SegmentationProbeHandler::add_label_overlays(
    NvDsBatchMeta* batch_meta,
    NvDsFrameMeta* frame_meta,
    const SegmentationTensorInfo& seg_info,
    int outW, int outH,
    cudaStream_t stream)
{
    // 3) Accumulate centroid stats on GPU
    if (accumulate_centroids_kernel_launch(gpu_buffers.class_map_dev,
                                          seg_info.W, seg_info.H, seg_info.C,
                                          gpu_buffers.count_dev,
                                          gpu_buffers.sumx_dev,
                                          gpu_buffers.sumy_dev,
                                          stream) != cudaSuccess) {
        return;
    }

    // 4) Copy centroid data back to host
    std::vector<int32_t> h_count(seg_info.C);
    std::vector<int64_t> h_sumx(seg_info.C), h_sumy(seg_info.C);

    cudaMemcpyAsync(h_count.data(), gpu_buffers.count_dev,
                   (size_t)seg_info.C * sizeof(int32_t),
                   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_sumx.data(), gpu_buffers.sumx_dev,
                   (size_t)seg_info.C * sizeof(int64_t),
                   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_sumy.data(), gpu_buffers.sumy_dev,
                   (size_t)seg_info.C * sizeof(int64_t),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5) Process centroids and create label items
    struct Item {
        int id;
        int cnt;
        int x;
        int y;
    };
    std::vector<Item> items;
    items.reserve(seg_info.C);

    for (int id = 1; id < seg_info.C; ++id) {
        int cnt = (int)h_count[id];
        if (cnt < context->config.visualization.min_pixels) continue;

        int cx = (int)(h_sumx[id] / (int64_t)cnt);
        int cy = (int)(h_sumy[id] / (int64_t)cnt);

        // Scale centroid to output NV12 coordinates
        int ox = (int)((int64_t)cx * outW / seg_info.W);
        int oy = (int)((int64_t)cy * outH / seg_info.H);

        ox = std::max(0, std::min(outW - 1, ox));
        oy = std::max(0, std::min(outH - 1, oy));
        items.push_back({id, cnt, ox, oy});
        
        // Save centroid for optical flow tracking
        gpu_buffers.last_centroids.push_back({id, (float)ox, (float)oy});
    }

    // Sort by pixel count (descending) and limit to top K
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.cnt > b.cnt; });
    if ((int)items.size() > context->config.visualization.top_k) {
        items.resize(context->config.visualization.top_k);
    }

    // 6) Create display metadata for labels
    if (!items.empty()) {
        NvDsDisplayMeta* dmeta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (dmeta) {
            dmeta->num_labels = 0;

            int max_slots = (int)(sizeof(dmeta->text_params) / sizeof(dmeta->text_params[0]));
            for (const auto& it : items) {
                if (dmeta->num_labels >= max_slots) break;

                NvOSD_TextParams& tp = dmeta->text_params[dmeta->num_labels];
                tp.display_text = g_strdup(context->class_names[it.id].c_str());
                tp.x_offset = it.x;
                tp.y_offset = it.y;

                tp.font_params.font_name = (gchar*)context->config.visualization.font.name.c_str();
                tp.font_params.font_size = context->config.visualization.font.size;
                tp.font_params.font_color = {
                    context->config.visualization.font.color_r,
                    context->config.visualization.font.color_g,
                    context->config.visualization.font.color_b,
                    context->config.visualization.font.color_a
                };

                tp.set_bg_clr = 1;
                tp.text_bg_clr = {
                    context->config.visualization.font.bg_r,
                    context->config.visualization.font.bg_g,
                    context->config.visualization.font.bg_b,
                    context->config.visualization.font.bg_a
                };

                dmeta->num_labels++;
            }

            nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
    }
}

GstPadProbeReturn SegmentationProbeHandler::handle_buffer(
    GstPad* pad,
    GstPadProbeInfo* info)
{
    (void)pad;

    if (!ensure_buffer_writable(info)) {
        return GST_PAD_PROBE_OK;
    }

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    GstMapInfo in_map{};
    NvBufSurface* surface = map_buffer(buf, in_map);
    if (!surface) return GST_PAD_PROBE_DROP;

    auto* pbm = find_preprocess_meta_for_uid(batch_meta,
                                            context->config.inference_ids.segmentation_uid);
    if (!pbm) {
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_DROP;
    }

    if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_DROP;
    }

    // Copy surface to ensure independence from other branches
    // The tee element shares the same NvBufSurface across branches
    surface = copy_and_replace_buffer_surface(buf, in_map, surface);
    if (!surface) {
        return GST_PAD_PROBE_DROP;
    }

    cudaStream_t stream = 0;
    // Lock metadata while modifying
    nvds_acquire_meta_lock(batch_meta);

    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
        if (!fmeta) continue;

        const int b = (int)fmeta->batch_id;
        if (b < 0 || b >= (int)surface->batchSize) continue;

        SegmentationTensorInfo seg_info;
        if (!extract_segmentation_tensor(batch_meta, fmeta, pbm, seg_info)) {
            // SKIPPED FRAME - Use Optical Flow Tracking
            if (gpu_buffers.last_class_map_dev && last_seg_info.C > 0) {
                NvDsOpticalFlowMeta* flow_meta = nullptr;
                for (NvDsMetaList* l = fmeta->frame_user_meta_list; l; l = l->next) {
                    NvDsUserMeta* user_meta = (NvDsUserMeta*)l->data;
                    if (user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META) {
                        flow_meta = (NvDsOpticalFlowMeta*)user_meta->user_meta_data;
                        break;
                    }
                }
                if (flow_meta && flow_meta->data) {
                    std::vector<TranslationVector> trans;
                    NvOFFlowVector* flow_data = (NvOFFlowVector*)flow_meta->data;
                    
                    NvBufSurfaceParams& sl = surface->surfaceList[b];
                    int outW = (int)sl.width;
                    int outH = (int)sl.height;
                    
                    for (auto& centroid : gpu_buffers.last_centroids) {
                        int block_size = 4;
                        int bx = (int)centroid.x / block_size;
                        int by = (int)centroid.y / block_size;
                        if (bx >= 0 && bx < (int)flow_meta->cols && by >= 0 && by < (int)flow_meta->rows) {
                            NvOFFlowVector mv = flow_data[by * flow_meta->cols + bx];
                            float dx_nv12 = mv.flowx / 32.0f;
                            float dy_nv12 = mv.flowy / 32.0f;
                            
                            // Map translation back to class_map resolution
                            int dx_map = (int)(dx_nv12 * last_seg_info.W / outW);
                            int dy_map = (int)(dy_nv12 * last_seg_info.H / outH);
                            
                            if (dx_map != 0 || dy_map != 0) {
                                trans.push_back({centroid.class_id, dx_map, dy_map});
                            }
                            
                            // Update centroid for next skipped frame
                            centroid.x += dx_nv12;
                            centroid.y += dy_nv12;
                        }
                    }
                    
                    if (!trans.empty()) {
                        TranslationVector* trans_dev;
                        cudaMalloc(&trans_dev, trans.size() * sizeof(TranslationVector));
                        cudaMemcpyAsync(trans_dev, trans.data(), trans.size() * sizeof(TranslationVector), cudaMemcpyHostToDevice, stream);
                        
                        translate_segmentation_masks_launch(
                            gpu_buffers.last_class_map_dev,
                            gpu_buffers.class_map_dev,
                            last_seg_info.W, last_seg_info.H,
                            trans_dev, trans.size(), stream);
                            
                        cudaFreeAsync(trans_dev, stream);
                        
                        // Copy current back to last for next iteration
                        cudaMemcpyAsync(gpu_buffers.last_class_map_dev, gpu_buffers.class_map_dev,
                                        gpu_buffers.class_map_bytes, cudaMemcpyDeviceToDevice, stream);
                    }
                    
                    // Render translated mask
                    int pitch = (int)sl.pitch;
                    uint8_t* base = (uint8_t*)sl.dataPtr;
                    uint8_t* y_dev = base;
                    uint8_t* uv_dev = base + (size_t)pitch * (size_t)outH;
                    seg_classmap_to_nv12_launch(gpu_buffers.class_map_dev, last_seg_info.W, last_seg_info.H,
                                                y_dev, uv_dev, outW, outH, pitch, pitch,
                                                context->config.visualization.alpha, stream);
                }
            }
            continue;
        }
        // keep a copy of the most recent tensor info for post‑loop processing
        last_seg_info = seg_info;

        if (!colorize_frame(surface, b, seg_info, stream)) {
            continue;
        }
        
        // Copy freshly inferred mask to last_class_map_dev
        cudaMemcpyAsync(gpu_buffers.last_class_map_dev, gpu_buffers.class_map_dev,
                        gpu_buffers.class_map_bytes, cudaMemcpyDeviceToDevice, stream);
                        
        // Clear old centroids for new inference
        gpu_buffers.last_centroids.clear();

        NvBufSurfaceParams& sl = surface->surfaceList[b];
        int outW = (int)sl.width;
        int outH = (int)sl.height;

        // add_label_overlays(batch_meta, fmeta, seg_info, outW, outH, stream);
    }

    // Clear object metadata to prevent nvdsosd from drawing detection boxes
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
        if (fmeta && fmeta->obj_meta_list) {
            nvds_clear_obj_meta_list(fmeta, fmeta->obj_meta_list);
            fmeta->obj_meta_list = nullptr;
        }
    }

    // Determine if this batch performed an actual segmentation inference
    gboolean infer = FALSE;
    gint interval = 0;
    GstElement *parent = GST_ELEMENT(gst_pad_get_parent(pad));
    if (parent) {
        g_object_get(G_OBJECT(parent), "interval", &interval, NULL);
        gst_object_unref(parent);
    }
    if (interval == 0) {
        infer = TRUE;
    } else {
        for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
            NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
            if (fmeta && (fmeta->frame_num % (interval + 1) == 0)) {
                infer = TRUE;
                break;
            }
        }
    }

    if (infer) {
        FPSTracker::getInstance().update("Segmentation(Infer)", batch_meta->num_frames_in_batch);

        // Only generate segmentation objects (bounding boxes) if we have valid inference metadata
        if (last_seg_info.C > 0 && last_seg_info.H > 0 && last_seg_info.W > 0) {
            // 1) Copy class map (GPU) → host
            int H = last_seg_info.H;
            int W = last_seg_info.W;
            int C = last_seg_info.C;
            size_t map_bytes = static_cast<size_t>(H) * W * sizeof(int32_t);
            std::vector<int32_t> h_class_map(H * W);
            cudaMemcpyAsync(h_class_map.data(), gpu_buffers.class_map_dev, map_bytes,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // 2) Build binary masks per label and generate NvDsObjectMeta
            std::map<int, cv::Mat> label_masks;
            cv::Mat class_map_mat(H, W, CV_32S, h_class_map.data());
            for (int lbl = 0; lbl < C; ++lbl) {
                cv::Mat mask = (class_map_mat == lbl);
                if (cv::countNonZero(mask) > 0) {
                    label_masks[lbl] = mask.clone();
                }
            }
            generate_segmentation_objects(batch_meta, label_masks);
        }
    }

    // 3) Through‑put FPS (already measured above)
    FPSTracker::getInstance().update("Segmentation", batch_meta->num_frames_in_batch);

    nvds_release_meta_lock(batch_meta);
    gst_buffer_unmap(buf, &in_map);
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn seg_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* handler = reinterpret_cast<SegmentationProbeHandler*>(user_data);
    return handler->handle_buffer(pad, info);
}

void segmentation_context_destroy(gpointer data) {
    auto* ctx = reinterpret_cast<SegmentationContext*>(data);
    delete ctx;
}
