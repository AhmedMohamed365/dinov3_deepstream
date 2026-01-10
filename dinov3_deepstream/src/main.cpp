#include "utils/gst_headers.h"
#include <cuda_runtime_api.h>


#include <cuda_fp16.h>

#include "utils_cuda/depth.h"
#include "utils_cuda/segmentation.h"
#include "utils/gst_utils.h"

#include <bits/stdc++.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static GstPadProbeReturn
dinov3_src_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
  (void)pad;

  // Configure these for your depth nvinfer
  const guint64 depth_uid = 2;              // gie-unique-id of depth nvinfer
  const guint64 detection_uid = 3;              // gie-unique-id of depth nvinfer
  const guint64 segmentation_uid = 4;              // gie-unique-id of segmentation
  const std::string depth_input_name = "features"; // must match depth model input layer name

  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  // If your pipeline has queues/converters, metadata can be copied; lock when modifying meta lists.
  nvds_acquire_meta_lock(batch_meta);

  if (already_has_preprocess_for_uids(batch_meta, {depth_uid, detection_uid, segmentation_uid})) {
    nvds_release_meta_lock(batch_meta);
    return GST_PAD_PROBE_OK;
  }

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
    if (!frame_meta) continue;

    // Find backbone tensor output meta on the frame
    NvDsInferTensorMeta *tensor_meta = nullptr;

    for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != nullptr; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (!user_meta) continue;

      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
        tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
        break;
      }
    }
    if (!tensor_meta) continue;
    if (!tensor_meta->num_output_layers) continue;
    if (!tensor_meta->out_buf_ptrs_dev) continue;

    // Choose which output layer to forward (by name "features", else first)
    int feat_idx = 0;
    for (unsigned i = 0; i < tensor_meta->num_output_layers; ++i) {
      const char *lname = tensor_meta->output_layers_info[i].layerName;
      if (lname && depth_input_name == lname) { feat_idx = (int)i; break; }
    }

    NvDsInferLayerInfo &layer = tensor_meta->output_layers_info[feat_idx];

    void *dev_ptr = tensor_meta->out_buf_ptrs_dev[feat_idx];
    if (!dev_ptr) continue;

    const size_t elem_sz = elem_size_from_infer_dtype(layer.dataType);
    const size_t vol = volume_from_dims(layer.inferDims);
    const size_t bytes = elem_sz ? (vol * elem_sz) : 0;

    // Build preprocess meta
    auto *pbm = new GstNvDsPreProcessBatchMeta();
    pbm->private_data = nullptr;
    pbm->target_unique_ids = {depth_uid, detection_uid, segmentation_uid};

    // ROI info (full-frame)
    NvDsRoiMeta roi_meta;
    std::memset(&roi_meta, 0, sizeof(roi_meta));
    roi_meta.roi.left   = 0;
    roi_meta.roi.top    = 0;
    roi_meta.roi.width  = frame_meta->pipeline_width;
    roi_meta.roi.height = frame_meta->pipeline_height;
    roi_meta.scale_ratio_x = 1.0f;
    roi_meta.scale_ratio_y = 1.0f;
    roi_meta.offset_left = 0;
    roi_meta.offset_top  = 0;
    roi_meta.frame_meta  = frame_meta;
    pbm->roi_vector.clear();
    pbm->roi_vector.push_back(roi_meta);

    // Tensor meta
    pbm->tensor_meta = new NvDsPreProcessTensorMeta();
    pbm->tensor_meta->raw_tensor_buffer = dev_ptr;      // IMPORTANT: GPU pointer
    pbm->tensor_meta->buffer_size       = bytes;
    pbm->tensor_meta->gpu_id            = tensor_meta->gpu_id; // if present in your build; otherwise set 0
    pbm->tensor_meta->data_type         = (layer.dataType == NvDsInferDataType::HALF)
                                          ? NvDsDataType_FP16
                                          : (layer.dataType == NvDsInferDataType::FLOAT ? NvDsDataType_FP32
                                                                                       : NvDsDataType_FP32);
    pbm->tensor_meta->tensor_name       = depth_input_name;   // MUST match depth input layer name
    pbm->tensor_meta->private_data      = nullptr;
    pbm->tensor_meta->meta_id           = 0;
    pbm->tensor_meta->maintain_aspect_ratio = FALSE;

    // tensor_shape: include batch dim, then layer dims
    pbm->tensor_meta->tensor_shape.clear();
    pbm->tensor_meta->tensor_shape.push_back(1);
    for (int d = 0; d < layer.inferDims.numDims; ++d) {
      pbm->tensor_meta->tensor_shape.push_back(layer.inferDims.d[d]);
    }

    // Attach at batch level
    NvDsUserMeta *bm_umeta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!bm_umeta) {
      delete pbm->tensor_meta;
      delete pbm;
      continue;
    }

    bm_umeta->user_meta_data = (void *)pbm;
    bm_umeta->base_meta.meta_type     = (NvDsMetaType)NVDS_PREPROCESS_BATCH_META;
    bm_umeta->base_meta.copy_func     = preprocess_batchmeta_copy_func;
    bm_umeta->base_meta.release_func  = preprocess_batchmeta_release_func;
    bm_umeta->base_meta.batch_meta    = batch_meta;

    nvds_add_user_meta_to_batch(batch_meta, bm_umeta);

    // std::cout << "[DINOv3->PreprocessMeta] frame=" << frame_meta->frame_num
    //           << " forwarded layer=" << (layer.layerName ? layer.layerName : "(null)")
    //           << " bytes=" << bytes
    //           << " to depth_gie_uid=" << depth_uid
    //           << " input_name=" << depth_input_name
    //           << "\n";

    break;
  }

  nvds_release_meta_lock(batch_meta);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
depth_src_pad_probe_cuda(GstPad* pad, GstPadProbeInfo* info, gpointer user_data)
{
  (void)pad; (void)user_data;

  constexpr guint DEPTH_UID = 2;
  const std::string DEPTH_LAYER = "depth";

  static uint64_t call_idx = 0;
  call_idx++;
  const bool dbg = (call_idx <= 10) || (call_idx % 120 == 0);

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  // Ensure buffer is not shared (we edit the surface in-place)
  if (!gst_buffer_is_writable(buf)) {
    GstBuffer* wbuf = gst_buffer_make_writable(buf);
    if (!wbuf) return GST_PAD_PROBE_OK;
    GST_PAD_PROBE_INFO_DATA(info) = wbuf;
    buf = wbuf;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  // Map GstBuffer -> NvBufSurface struct (CPU-visible descriptor; NOT mapping GPU memory)
  GstMapInfo in_map{};
  if (!gst_buffer_map(buf, &in_map, GST_MAP_READ)) return GST_PAD_PROBE_OK;
  NvBufSurface* surface = (NvBufSurface*)in_map.data;
  if (!surface) { gst_buffer_unmap(buf, &in_map); return GST_PAD_PROBE_OK; }

  if (dbg) {
    std::cout << "\n[DEPTH/CUDA] call #" << call_idx
              << " memType=" << (int)surface->memType
              << " numFilled=" << surface->numFilled
              << " batchSize=" << surface->batchSize
              << "\n";
  }

  // This is your helper that finds the preprocess meta targeting uid=2
  auto* pbm = find_preprocess_meta_for_uid(batch_meta, DEPTH_UID);
  if (!pbm) {
    if (dbg) std::cout << "[DEPTH/CUDA] no NVDS_PREPROCESS_BATCH_META for uid=2\n";
    gst_buffer_unmap(buf, &in_map);
    return GST_PAD_PROBE_OK;
  }

  int updated = 0;

  for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
    if (!fmeta) continue;

    const int b = (int)fmeta->batch_id;
    if (b < 0 || b >= (int)surface->batchSize) continue;

    // --- Find depth tensor meta under ROI user meta (preprocessed mode) ---
    NvDsInferTensorMeta* depth_tmeta = nullptr;
    for (size_t r = 0; r < pbm->roi_vector.size(); ++r) {
      auto& roi_meta = pbm->roi_vector[r];
      depth_tmeta = find_tensor_meta_in_user_meta_list(roi_meta.roi_user_meta_list, DEPTH_UID);
      if (depth_tmeta) break;
    }
    if (!depth_tmeta) {
      if (dbg) std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num << " no depth tensor meta\n";
      continue;
    }

    int li = find_layer_index(depth_tmeta, DEPTH_LAYER);
    if (li < 0) {
      if (dbg) std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num
                         << " depth layer '" << DEPTH_LAYER << "' not found\n";
      continue;
    }

    NvDsInferLayerInfo& layer = depth_tmeta->output_layers_info[li];

    // Expect float depth on GPU
    if (layer.dataType != NvDsInferDataType::FLOAT) {
      if (dbg) std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num
                         << " depth dtype not FLOAT\n";
      continue;
    }

    void* depth_dev_void = (depth_tmeta->out_buf_ptrs_dev ? depth_tmeta->out_buf_ptrs_dev[li] : nullptr);
    if (!depth_dev_void) {
      if (dbg) std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num << " no out_buf_ptrs_dev\n";
      continue;
    }
    const float* depth_dev = (const float*)depth_dev_void;

    int H = 0, W = 0;
    if (layer.inferDims.numDims >= 2) {
      H = layer.inferDims.d[layer.inferDims.numDims - 2];
      W = layer.inferDims.d[layer.inferDims.numDims - 1];
    }
    if (H <= 0 || W <= 0) continue;

    // --- Get NV12 surface device pointers ---
    NvBufSurfaceParams& sl = surface->surfaceList[b];

    if (sl.colorFormat != NVBUF_COLOR_FORMAT_NV12 &&
        sl.colorFormat != NVBUF_COLOR_FORMAT_NV12_ER) {
      if (dbg) std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num
                         << " surface not NV12 (fmt=" << (int)sl.colorFormat << ")\n";
      continue;
    }

    if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
      if (dbg) std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num
                         << " expected NVBUF_MEM_CUDA_DEVICE, got memType=" << (int)surface->memType << "\n";
      continue;
    }

    // NV12 layout (pitch-linear, block-linear=false in your caps):
    // Y plane at dataPtr
    // UV plane at dataPtr + pitch * height
    // (If your DeepStream version exposes plane offsets, use them instead.)
    uint8_t* base = (uint8_t*)sl.dataPtr;       // device pointer
    int pitchY = (int)sl.pitch;                 // bytes per row
    int pitchUV = pitchY;
    int outW = (int)sl.width;
    int outH = (int)sl.height;

    uint8_t* y_dev  = base;
    uint8_t* uv_dev = base + (size_t)pitchY * (size_t)outH;

    // --- Launch CUDA kernels ---
    // Pick a fixed visualization range (meters). Your logs show ~0.8..4.4.
    // Tune as you like, or make them parameters.
    float near_m = 0.5f;
    float far_m  = 4.0f;

    cudaStream_t stream = 0; // default stream (works; later you can optimize)

    cudaError_t e = depth_to_nv12_colormap_launch(
        depth_dev, W, H,
        y_dev, uv_dev,
        outW, outH, pitchY, pitchUV,
        near_m, far_m,
        stream);

    if (e != cudaSuccess) {
      if (dbg) std::cout << "[DEPTH/CUDA] launch error: " << cudaGetErrorString(e) << "\n";
      continue;
    }

    cudaError_t e2 = cudaStreamSynchronize(stream); // correctness first
    if (e2 != cudaSuccess) {
      if (dbg) std::cout << "[DEPTH/CUDA] sync error: " << cudaGetErrorString(e2) << "\n";
      continue;
    }

    if (dbg) {
      std::cout << "[DEPTH/CUDA] frame=" << fmeta->frame_num
                << " wrote NV12 from depth " << W << "x" << H
                << " -> " << outW << "x" << outH
                << " pitch=" << pitchY
                << " near=" << near_m << " far=" << far_m
                << "\n";
    }

    updated++;
  }

  if (dbg && updated == 0) std::cout << "[DEPTH/CUDA] no frames updated\n";

  gst_buffer_unmap(buf, &in_map);
  return GST_PAD_PROBE_OK;
}


// Overwrite the NV12 frame with a colorized segmentation map (GPU-only).
static GstPadProbeReturn
seg_src_pad_probe_cuda(GstPad* pad, GstPadProbeInfo* info, gpointer user_data)
{
  (void)pad; (void)user_data;

  constexpr guint SEG_UID = 4;
  const std::string SEG_LAYER = "semantic_segmentation";
  constexpr float ALPHA = 0.5f; // 1.0 = pure seg colors; <1 blends with existing NV12

  static int32_t* class_map_dev = nullptr;
  static size_t   class_map_dev_bytes = 0;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  // We will overwrite the surface in-place -> ensure writable
  if (!gst_buffer_is_writable(buf)) {
    GstBuffer* wbuf = gst_buffer_make_writable(buf);
    if (!wbuf) return GST_PAD_PROBE_OK;
    GST_PAD_PROBE_INFO_DATA(info) = wbuf;
    buf = wbuf;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  // Map GstBuffer -> NvBufSurface* descriptor (not mapping GPU memory)
  GstMapInfo in_map{};
  if (!gst_buffer_map(buf, &in_map, GST_MAP_READ)) return GST_PAD_PROBE_OK;
  NvBufSurface* surface = reinterpret_cast<NvBufSurface*>(in_map.data);
  if (!surface) { gst_buffer_unmap(buf, &in_map); return GST_PAD_PROBE_OK; }

  // Find preprocess meta that targets SEG_UID (your dinov3_src_pad_probe attaches this)
  auto* pbm = find_preprocess_meta_for_uid(batch_meta, SEG_UID);
  if (!pbm) { gst_buffer_unmap(buf, &in_map); return GST_PAD_PROBE_OK; }

  // We expect CUDA-device NV12 surfaces to overwrite directly
  if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
    gst_buffer_unmap(buf, &in_map);
    return GST_PAD_PROBE_OK;
  }

  cudaStream_t stream = 0;

  for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    NvDsFrameMeta* fmeta = reinterpret_cast<NvDsFrameMeta*>(l_frame->data);
    if (!fmeta) continue;

    const int b = (int)fmeta->batch_id;
    if (b < 0 || b >= (int)surface->batchSize) continue;

    // Find tensor meta produced by seg nvinfer under ROI user meta
    NvDsInferTensorMeta* tmeta = nullptr;
    for (size_t r = 0; r < pbm->roi_vector.size(); ++r) {
      auto& roi_meta = pbm->roi_vector[r];
      tmeta = find_tensor_meta_in_user_meta_list(roi_meta.roi_user_meta_list, SEG_UID);
      if (tmeta) break;
    }
    if (!tmeta) continue;

    int li = find_layer_index(tmeta, SEG_LAYER);
    if (li < 0) continue;

    NvDsInferLayerInfo& layer = tmeta->output_layers_info[li];
    void* logits_dev = (tmeta->out_buf_ptrs_dev ? tmeta->out_buf_ptrs_dev[li] : nullptr);
    if (!logits_dev) continue;

    // Expect [1,C,H,W] (NCHW) or [C,H,W]
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
      continue;
    }
    if (C <= 0 || H <= 0 || W <= 0) continue;

    const bool is_half = (layer.dataType == NvDsInferDataType::HALF);
    if (!(layer.dataType == NvDsInferDataType::FLOAT || layer.dataType == NvDsInferDataType::HALF))
      continue;

    // Allocate / grow GPU class map buffer (HxW int32)
    const size_t n_pix = (size_t)H * (size_t)W;
    const size_t need_bytes = n_pix * sizeof(int32_t);
    if (!class_map_dev || class_map_dev_bytes < need_bytes) {
      if (class_map_dev) cudaFree(class_map_dev);
      cudaMalloc(&class_map_dev, need_bytes);
      class_map_dev_bytes = need_bytes;
    }

    // 1) Argmax on GPU: logits -> class_map_dev
    cudaError_t e1 = seg_argmax_launch(logits_dev, is_half, C, H, W, class_map_dev, stream);
    if (e1 != cudaSuccess) continue;

    // 2) Overwrite NV12 surface with colorized seg
    NvBufSurfaceParams& sl = surface->surfaceList[b];
    if (sl.colorFormat != NVBUF_COLOR_FORMAT_NV12 &&
        sl.colorFormat != NVBUF_COLOR_FORMAT_NV12_ER) {
      continue;
    }

    const int outW = (int)sl.width;
    const int outH = (int)sl.height;
    const int pitchY = (int)sl.pitch;
    const int pitchUV = pitchY;

    uint8_t* base = reinterpret_cast<uint8_t*>(sl.dataPtr);   // device pointer
    if (!base) continue;

    uint8_t* y_dev  = base;
    uint8_t* uv_dev = base + (size_t)pitchY * (size_t)outH;   // NV12 UV plane after Y plane

    cudaError_t e2 = seg_classmap_to_nv12_launch(
        class_map_dev, W, H,
        y_dev, uv_dev,
        outW, outH, pitchY, pitchUV,
        ALPHA, stream);
    if (e2 != cudaSuccess) continue;

    // correctness first (remove or replace with async handling once stable)
    cudaStreamSynchronize(stream);
  }

  gst_buffer_unmap(buf, &in_map);
  return GST_PAD_PROBE_OK;
}



static GstPadProbeReturn
nvsegvisual_sink_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data)
{
  (void)pad; (void)user_data;
  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
    if (!fmeta) continue;

    for (NvDsMetaList* l = fmeta->frame_user_meta_list; l; l = l->next) {
      NvDsUserMeta* um = (NvDsUserMeta*)l->data;
      if (!um) continue;
      if (um->base_meta.meta_type != NVDSINFER_SEGMENTATION_META) continue;

      auto* sm = (NvDsInferSegmentationMeta*)um->user_meta_data;
      if (!sm || !sm->class_map) continue;

      // sample min/max quickly (don’t scan all pixels every frame)
      int mn = 1e9, mx = -1e9;
      int step = (sm->width * sm->height) / 2000; // ~2000 samples
      if (step < 1) step = 1;
      for (guint i = 0; i < sm->width * sm->height; i += step) {
        int v = sm->class_map[i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }

      std::cout << "[NVSEGVISUAL SINK] frame=" << fmeta->frame_num
                << " uid=" << sm->unique_id
                << " classes=" << sm->classes
                << " size=" << sm->width << "x" << sm->height
                << " sample_min=" << mn << " sample_max=" << mx
                << "\n";
      break;
    }
  }
  return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
  std::string device = "/dev/video0";
  std::string infer_cfg = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_dinov3.txt";
  std::string depth_cfg = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_depth.txt";
  std::string detection_cfg = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_detection.txt";
  std::string segmentation_cfg = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_segmentation.txt";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--device" && i + 1 < argc) device = argv[++i];
    else if (a == "--config" && i + 1 < argc) infer_cfg = argv[++i];
    else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << " [--device /dev/video0] [--config configs/config_infer_dinov3.txt]\n";
      return 0;
    }
  }

  gst_init(&argc, &argv);

  // Pipeline:
  // v4l2src -> videoconvert -> nvvideoconvert -> NVMM -> nvstreammux -> nvinfer -> nvvideoconvert -> nveglglessink
  //
  // Note: nvstreammux requires linking to mux.sink_0
  std::string pipeline_desc =
  "v4l2src device=" + device + " ! "
  "video/x-raw,framerate=30/1 ! "
  "videoconvert ! "
  "video/x-raw,format=RGBA ! "
  "nvvideoconvert ! "
  "video/x-raw(memory:NVMM),format=NV12 ! "
  "queue ! mux.sink_0 "
  "nvstreammux name=mux batch-size=1 width=640 height=640 live-source=1 batched-push-timeout=40000 ! "
  "tee name=t0 "
  "t0. ! queue ! nvvideoconvert ! nveglglessink sync=false "
  "t0. ! queue ! "
    "nvinfer name=dinov3 config-file-path=" + infer_cfg + " ! tee name=t1 "
      // "t1. ! queue ! nvinfer name=depth config-file-path=" + depth_cfg + " ! "
      // "nvvideoconvert name=postdepthconv ! "
      // "video/x-raw(memory:NVMM),format=NV12 ! "
      // "nveglglessink sync=false "
      // "t1. ! queue ! nvinfer name=detection config-file-path=" + detection_cfg + " ! "
      // "nvvideoconvert name=postdetectionconv ! "
      // "video/x-raw(memory:NVMM),format=RGBA ! nvdsosd ! "
      // "nveglglessink sync=false "
      "t1. ! queue ! nvinfer name=seg config-file-path=" + segmentation_cfg + " ! "
      "nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA,width=640,height=640 ! "
      "nveglglessink sync=false "
      ;

  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
  if (!pipeline) {
    std::cerr << "Failed to create pipeline.\n";
    if (error) std::cerr << "Error: " << error->message << "\n";
    return 1;
  }

  // Attach probe on dinov3 src pad
  GstElement *dinov3 = gst_bin_get_by_name(GST_BIN(pipeline), "dinov3");
  if (!dinov3) {
    std::cerr << "Could not find nvinfer element named 'dinov3'\n";
    gst_object_unref(pipeline);
    return 1;
  }

  GstPad *srcpad = gst_element_get_static_pad(dinov3, "src");
  if (!srcpad) {
    std::cerr << "Could not get src pad of 'dinov3'\n";
    gst_object_unref(dinov3);
    gst_object_unref(pipeline);
    return 1;
  }

  gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_BUFFER, dinov3_src_pad_probe, nullptr, nullptr);
  gst_object_unref(srcpad);
  gst_object_unref(dinov3);

  // GstElement* depth = gst_bin_get_by_name(GST_BIN(pipeline), "depth");
  // if (!depth) {
  //   std::cerr << "Could not find nvinfer element named 'depth'\n";
  //   gst_object_unref(pipeline);
  //   return 1;
  // }
  // GstPad* depth_srcpad = gst_element_get_static_pad(depth, "src");
  // if (!depth_srcpad) {
  //   std::cerr << "Could not get src pad of 'depth'\n";
  //   gst_object_unref(depth);
  //   gst_object_unref(pipeline);
  //   return 1;
  // }
  // gst_pad_add_probe(depth_srcpad, GST_PAD_PROBE_TYPE_BUFFER, depth_src_pad_probe_cuda, nullptr, nullptr);
  // gst_object_unref(depth_srcpad);
  // gst_object_unref(depth);


  GstElement* seg = gst_bin_get_by_name(GST_BIN(pipeline), "seg");
  if (!seg) {
    std::cerr << "Could not find nvinfer element named 'seg'\n";
    return 1;
  }
  GstPad* seg_srcpad = gst_element_get_static_pad(seg, "src");
  gst_pad_add_probe(seg_srcpad, GST_PAD_PROBE_TYPE_BUFFER, seg_src_pad_probe_cuda, nullptr, nullptr);
  gst_object_unref(seg_srcpad);
  gst_object_unref(seg);

  // Run
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  GstBus *bus = gst_element_get_bus(pipeline);
  bool running = true;
  while (running) {
    GstMessage *msg = gst_bus_timed_pop_filtered(
        bus, GST_CLOCK_TIME_NONE,
        (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (!msg) continue;

    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError *err = nullptr;
        gchar *dbg = nullptr;
        gst_message_parse_error(msg, &err, &dbg);
        std::cerr << "GStreamer ERROR: " << (err ? err->message : "unknown") << "\n";
        if (dbg) std::cerr << "Debug: " << dbg << "\n";
        if (err) g_error_free(err);
        if (dbg) g_free(dbg);
        running = false;
        break;
      }
      case GST_MESSAGE_EOS:
        running = false;
        break;
      default:
        break;
    }
    gst_message_unref(msg);
  }

  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}