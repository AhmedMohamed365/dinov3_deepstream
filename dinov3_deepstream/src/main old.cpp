#include <gst/gst.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsmeta.h"
#include "nvdspreprocess_meta.h"


#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static GstPadProbeReturn
dinov3_src_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    // Look for NVDSINFER_TENSOR_OUTPUT_META on frame_user_meta_list
    for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != nullptr; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (!user_meta) continue;

      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
        auto *tensor_meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;

        std::cout << "[DINOv3] frame=" << frame_meta->frame_num
                  << " outputs=" << tensor_meta->num_output_layers
                  << " (unique_id=" << tensor_meta->unique_id << ")\n";

        for (unsigned i = 0; i < tensor_meta->num_output_layers; ++i) {
          NvDsInferLayerInfo &layer = tensor_meta->output_layers_info[i];

          std::cout << "  - layer[" << i << "] name=" << (layer.layerName ? layer.layerName : "(null)")
                    << " dims=";

          // dims is NvDsInferDims
          for (int d = 0; d < layer.inferDims.numDims; ++d) {
            std::cout << layer.inferDims.d[d] << (d + 1 < layer.inferDims.numDims ? "x" : "");
          }

          // Try to print a couple values if host buffer exists
          if (tensor_meta->out_buf_ptrs_host && tensor_meta->out_buf_ptrs_host[i]) {
            // Assume float output for embeddings/features (common for backbones)
            float *p = (float *) tensor_meta->out_buf_ptrs_host[i];
            std::cout << "  sample=[" << p[0] << ", " << p[1] << ", " << p[2] << "]";
          } else {
            std::cout << "  (no host buf)";
          }

          std::cout << "\n";
        }
      }
    }
  }

  return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
  std::string device = "/dev/video0";
  std::string infer_cfg = "/dinov3_deepstream/dinov3_deepstream/configs/config_infer_dinov3.txt";

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
      "nvstreammux name=mux batch-size=1 width=800 height=800 live-source=1 batched-push-timeout=40000 ! "
      "nvinfer name=dinov3 config-file-path=" + infer_cfg + " ! "
      "nvvideoconvert ! "
      "nveglglessink sync=false";

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