#include "utils/gst_headers.h"
#include <cuda_runtime_api.h>


#include <cuda_fp16.h>

#include "utils_cuda/depth.h"
#include "utils_cuda/segmentation.h"
#include "utils/gst_utils.h"
#include "utils/file_utils.h"
#include "config/app_config.h"
#include "probes/dinov3_probe.h"
#include "probes/depth_probe.h"
#include "probes/segmentation_probe.h"
#include "pipeline/pipeline_builder.h"

#include <bits/stdc++.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  // Create default configuration
  AppConfig app_config = AppConfig::create_default();

  // Parse command-line arguments to override defaults
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--device" && i + 1 < argc) app_config.pipeline.device = argv[++i];
    else if (a == "--config" && i + 1 < argc) app_config.model_paths.dinov3_config = argv[++i];
    else if (a == "--do-depth" && i + 1 < argc) {
      std::string val = argv[++i];
      app_config.inference_enable.depth = (val == "true" || val == "1");
    }
    else if (a == "--do-detection" && i + 1 < argc) {
      std::string val = argv[++i];
      app_config.inference_enable.detection = (val == "true" || val == "1");
    }
    else if (a == "--do-segmentation" && i + 1 < argc) {
      std::string val = argv[++i];
      app_config.inference_enable.segmentation = (val == "true" || val == "1");
    }
    else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                << "Options:\n"
                << "  --device DEVICE               Video device path (default: /dev/video0)\n"
                << "  --config CONFIG               DINOv3 config file path\n"
                << "  --do-depth [true|false]       Enable/disable depth estimation (default: true)\n"
                << "  --do-detection [true|false]   Enable/disable object detection (default: true)\n"
                << "  --do-segmentation [true|false] Enable/disable segmentation (default: true)\n"
                << "  -h, --help                    Show this help message\n";
      return 0;
    }
  }

  // Convenience aliases for cleaner code
  std::string device = app_config.pipeline.device;
  std::string infer_cfg = app_config.model_paths.dinov3_config;
  std::string depth_cfg = app_config.model_paths.depth_config;
  std::string detection_cfg = app_config.model_paths.detection_config;
  std::string segmentation_cfg = app_config.model_paths.segmentation_config;
  std::string labels_segmentation = app_config.model_paths.segmentation_labels;

  // Conditionally create segmentation context
  SegmentationContext* seg_ctx = nullptr;
  if (app_config.inference_enable.segmentation) {
    seg_ctx = new SegmentationContext();
    seg_ctx->config = app_config;
    try {
      seg_ctx->class_names = load_lines_txt(labels_segmentation);
      std::cout << "[INFO] Loaded " << seg_ctx->class_names.size()
                << " class names from " << labels_segmentation << "\n";
    } catch (const std::exception& e) {
      std::cerr << "[WARN] " << e.what() << " (will use fallback class_<id>)\n";
    }
  }

  gst_init(&argc, &argv);

  // Build and create pipeline
  PipelineBuilder builder(app_config);
  GError* error = nullptr;
  GstElement* pipeline = builder.create_pipeline(&error);
  if (!pipeline) {
    std::cerr << "Failed to create pipeline.\n";
    if (error) std::cerr << "Error: " << error->message << "\n";
    return 1;
  }

  // Attach DINOv3 probe (always - backbone required)
  auto* dinov3_handler = new DINOv3ProbeHandler(app_config);
  if (!PipelineProbeAttacher::attach_probe_to_element(
          pipeline, "dinov3",
          dinov3_src_pad_probe_wrapper,
          dinov3_handler,
          [](gpointer data) { delete reinterpret_cast<DINOv3ProbeHandler*>(data); })) {
    gst_object_unref(pipeline);
    return 1;
  }

  // Conditionally attach depth probe
  if (app_config.inference_enable.depth) {
    auto* depth_handler = new DepthProbeHandler(app_config);
    if (!PipelineProbeAttacher::attach_probe_to_element(
            pipeline, "depth",
            depth_src_pad_probe_wrapper,
            depth_handler,
            [](gpointer data) { delete reinterpret_cast<DepthProbeHandler*>(data); })) {
      gst_object_unref(pipeline);
      return 1;
    }
  }

  // Conditionally attach segmentation probe
  if (app_config.inference_enable.segmentation) {
    auto* seg_handler = new SegmentationProbeHandler(seg_ctx);
    if (!PipelineProbeAttacher::attach_probe_to_element(
            pipeline, "seg",
            seg_src_pad_probe_wrapper,
            seg_handler,
            [](gpointer data) {
              auto* handler = reinterpret_cast<SegmentationProbeHandler*>(data);
              auto* ctx = handler->context;
              delete handler;
              delete ctx;
            })) {
      gst_object_unref(pipeline);
      return 1;
    }
  }

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