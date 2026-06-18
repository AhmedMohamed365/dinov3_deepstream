#include "pipeline_builder.h"
#include <iostream>
#include <sstream>
#include <cmath>

std::string PipelineBuilder::build_source_branch() {
    std::stringstream ss;

    switch (config.pipeline.source_type) {
        case SourceType::CAMERA:
            // USB/V4L2 camera source
            ss << "v4l2src device=" << config.pipeline.source_uri << " ! "
               << "videoconvert ! "
               << "videorate ! "
               << "video/x-raw,format=RGBA,framerate=" << config.pipeline.framerate << "/1 ! "
               << "nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;

        case SourceType::FILE:
            ss << "nvurisrcbin uri=file://" << config.pipeline.source_uri
               << " disable-audio=true file-loop=true"
               << " ! mux.sink_0 ";
            break;

        case SourceType::RTSP:
            // RTSP stream source
            ss << "nvurisrcbin uri=" << config.pipeline.source_uri
               << " latency=200"
               << " rtsp-reconnect-interval=30"
               << " rtsp-reconnect-attempts=-1"
               << " disable-audio=true"
               << " ! queue ! "
               << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;

        case SourceType::URI:
            ss << "nvurisrcbin uri=" << config.pipeline.source_uri
               << " disable-audio=true "
               << "! queue ! "
               << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;
    }

    return ss.str();
}

std::string PipelineBuilder::build_muxer_config() {
    std::stringstream ss;
    ss << "nvstreammux name=mux "
       << "batch-size=" << config.pipeline.batch_size << " "
       << "width=" << config.pipeline.width << " "
       << "height=" << config.pipeline.height << " "
       << "live-source=" << config.pipeline.live_source << " "
       << "batched-push-timeout=" << config.pipeline.batched_push_timeout << " ! ";
    return ss.str();
}

std::string PipelineBuilder::build_tiled_muxer() {
    std::stringstream ss;

    // Count enabled heads to configure tiler (including original image)
    int num_tiles = 1; // Start with 1 for original image
    if (config.inference_enable.depth) num_tiles++;
    if (config.inference_enable.detection) num_tiles++;
    if (config.inference_enable.segmentation) num_tiles++;
    if (config.inference_enable.optical_flow) num_tiles++;

    // Calculate rows and columns for tiling (prefer square-ish layout)
    int cols = (int)std::ceil(std::sqrt(num_tiles));
    int rows = (int)std::ceil((double)num_tiles / cols);

    // Create a muxer for the tiler with reduced timeout to minimize lag
    ss << "nvstreammux name=tilemux "
       << "batch-size=" << num_tiles << " "
       << "width=" << config.pipeline.width << " "
       << "height=" << config.pipeline.height << " "
       << "live-source=" << config.pipeline.live_source << " "
       << "batched-push-timeout=4000 ! ";  // Reduced from 40000 to 4000 (4ms)

    // Add tiler and single sink
    ss << "nvmultistreamtiler "
       << "rows=" << rows << " "
       << "columns=" << cols << " "
       << "width=" << (config.pipeline.width * cols) << " "
       << "height=" << (config.pipeline.height * rows) << " ! "
       << "nvvideoconvert ! ";

    if (config.pipeline.rtsp_output) {
        ss << "video/x-raw(memory:NVMM),format=I420 ! "
           << "nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 ! "
           << "h264parse ! "
           << "rtph264pay config-interval=1 ! "
           << "udpsink host=127.0.0.1 port=5400 sync="
           << (config.pipeline.live_source ? "false" : "true")
           << " async=false ";
    } else {
        ss << "nveglglessink sync=" << (config.pipeline.live_source ? "false" : "true") << " ";
    }

    return ss.str();
}

std::string PipelineBuilder::build_visualization_branch() {
    std::stringstream ss;
    ss << "tee name=t0 ";

    if (config.pipeline.rtsp_output) {
        if (config.pipeline.display_mode == DisplayMode::TILED) {
            ss << build_tiled_muxer();
            ss << "t0. ! queue name=q_original ! "
               << "nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=RGBA ! "
               << "tilemux.sink_0 ";
        } else {
            // SEPARATE mode (RTSP): Stream original image to port 5400
            ss << "t0. ! queue ! nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=I420 ! "
               << "nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 ! "
               << "h264parse ! "
               << "rtph264pay config-interval=1 ! "
               << "udpsink host=127.0.0.1 port=5400 sync="
               << (config.pipeline.live_source ? "false" : "true")
               << " async=false ";
        }
    } else if (config.pipeline.display_mode == DisplayMode::SEPARATE) {
        // SEPARATE mode: create visualization window
        ss << "t0. ! queue ! nvvideoconvert ! nveglglessink sync="
           << (config.pipeline.live_source ? "false" : "true") << " ";
    } else {
        // TILED mode: create tilemux and connect original image as first tile
        ss << build_tiled_muxer();
        ss << "t0. ! queue name=q_original ! "
           << "nvvideoconvert ! "
           << "video/x-raw(memory:NVMM),format=RGBA ! "
           << "tilemux.sink_0 ";
    }

    return ss.str();
}

std::string PipelineBuilder::build_inference_branches() {
    std::stringstream ss;

    // Shared DINOv3 backbone (runs once)
    ss << "t0. ! queue ! "
       << "nvinfer name=dinov3 config-file-path=" << config.model_paths.dinov3_config
       << " ! tee name=t1 allow-not-linked=true ";

    int sink_idx = 1;

    // Depth branch
    if (config.inference_enable.depth) {
        ss << "t1. ! queue name=q_depth ! "
           << "nvinfer name=depth config-file-path="
           << config.model_paths.depth_config << " ! "
           << "nvvideoconvert name=postdepthconv ! "
           << "video/x-raw(memory:NVMM),format=RGBA ! "
           << "tee name=t_depth ";

        // Branch 1: Connect to tilemux if tiled mode is active
        if (config.pipeline.display_mode == DisplayMode::TILED) {
            ss << "t_depth. ! queue ! tilemux.sink_" << sink_idx++ << " ";
        }

        // Branch 2: Handle main output (RTSP or local separate window)
        if (config.pipeline.rtsp_output) {
            // Stream to UDP port 5401 for /depth
            ss << "t_depth. ! queue ! nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=I420 ! "
               << "nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 ! "
               << "h264parse ! "
               << "rtph264pay config-interval=1 ! "
               << "udpsink host=127.0.0.1 port=5401 sync="
               << (config.pipeline.live_source ? "false" : "true")
               << " async=false ";
        } else {
            // Local display mode
            if (config.pipeline.display_mode == DisplayMode::SEPARATE) {
                ss << "t_depth. ! queue ! nveglglessink sync=false ";
            } else {
                ss << "t_depth. ! queue ! fakesink sync=false ";
            }
        }
    }

    // Detection branch
    if (config.inference_enable.detection) {
        ss << "t1. ! queue name=q_det ! "
           << "nvinfer name=detection config-file-path="
           << config.model_paths.detection_config << " ! ";

        if (config.pipeline.enable_tracker) {
            ss << "nvtracker name=tracker "
               << "ll-config-file=" << config.pipeline.tracker_config << " "
               << "ll-lib-file=" << config.pipeline.tracker_lib << " "
               << "tracker-width=640 tracker-height=640 gpu-id=0 ! ";
        }

        ss << "nvvideoconvert name=postdetectionconv ! "
           << "video/x-raw(memory:NVMM),format=RGBA ! nvdsosd ! "
           << "tee name=t_det ";

        // Branch 1: Connect to tilemux if tiled mode is active
        if (config.pipeline.display_mode == DisplayMode::TILED) {
            ss << "t_det. ! queue ! tilemux.sink_" << sink_idx++ << " ";
        }

        // Branch 2: Handle main output (RTSP or local separate window)
        if (config.pipeline.rtsp_output) {
            // Stream to UDP port 5402 for /detection
            ss << "t_det. ! queue ! nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=I420 ! "
               << "nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 ! "
               << "h264parse ! "
               << "rtph264pay config-interval=1 ! "
               << "udpsink host=127.0.0.1 port=5402 sync="
               << (config.pipeline.live_source ? "false" : "true")
               << " async=false ";
        } else {
            // Local display mode
            if (config.pipeline.display_mode == DisplayMode::SEPARATE) {
                ss << "t_det. ! queue ! nveglglessink sync=false ";
            } else {
                ss << "t_det. ! queue ! fakesink sync=false ";
            }
        }
    }

    // Segmentation branch
    if (config.inference_enable.segmentation) {
        ss << "t1. ! queue name=q_seg ! "
           << "nvinfer name=seg config-file-path=" << config.model_paths.segmentation_config << " ! ";
        // optional tracker for segmentation objects
        if (config.pipeline.enable_tracker) {
            ss << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! nvof ! "
               << "nvtracker name=seg_tracker "
               << "ll-config-file=" << config.pipeline.tracker_config << " "
               << "ll-lib-file=" << config.pipeline.tracker_lib << " "
               << "tracker-width=640 tracker-height=640 gpu-id=0 ! ";
        }
        ss << "nvvideoconvert ! "
           << "video/x-raw(memory:NVMM),format=RGBA,width=" << config.pipeline.width
           << ",height=" << config.pipeline.height << " ! nvdsosd ! "
           << "tee name=t_seg ";

        // Branch 1: Connect to tilemux if tiled mode is active
        if (config.pipeline.display_mode == DisplayMode::TILED) {
            ss << "t_seg. ! queue ! tilemux.sink_" << sink_idx++ << " ";
        }

        // Branch 2: Handle main output (RTSP or local separate window)
        if (config.pipeline.rtsp_output) {
            // Stream to UDP port 5403 for /segmentation
            ss << "t_seg. ! queue ! nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=I420 ! "
               << "nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 ! "
               << "h264parse ! "
               << "rtph264pay config-interval=1 ! "
               << "udpsink host=127.0.0.1 port=5403 sync="
               << (config.pipeline.live_source ? "false" : "true")
               << " async=false ";
        } else {
            // Local display mode
            if (config.pipeline.display_mode == DisplayMode::SEPARATE) {
                ss << "t_seg. ! queue ! nveglglessink sync=false ";
            } else {
                ss << "t_seg. ! queue ! fakesink sync=false ";
            }
        }
    }

    // Optical flow branch
    if (config.inference_enable.optical_flow) {
        ss << "t1. ! queue name=q_flow ! "
           << "nvinfer name=optical_flow config-file-path="
           << config.model_paths.optical_flow_config << " ! "
           << "nvvideoconvert name=postflowconv ! "
           << "video/x-raw(memory:NVMM),format=RGBA ! "
           << "tee name=t_flow ";

        // Branch 1: Connect to tilemux if tiled mode is active
        if (config.pipeline.display_mode == DisplayMode::TILED) {
            ss << "t_flow. ! queue ! tilemux.sink_" << sink_idx++ << " ";
        }

        // Branch 2: Handle main output (RTSP or local separate window)
        if (config.pipeline.rtsp_output) {
            // Stream to UDP port 5404 for /optical-flow
            ss << "t_flow. ! queue ! nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=I420 ! "
               << "nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 ! "
               << "h264parse ! "
               << "rtph264pay config-interval=1 ! "
               << "udpsink host=127.0.0.1 port=5404 sync="
               << (config.pipeline.live_source ? "false" : "true")
               << " async=false ";
        } else {
            // Local display mode
            if (config.pipeline.display_mode == DisplayMode::SEPARATE) {
                ss << "t_flow. ! queue ! nveglglessink sync=false ";
            } else {
                ss << "t_flow. ! queue ! fakesink sync=false ";
            }
        }
    }

    return ss.str();
}

std::string PipelineBuilder::build_pipeline_description() {
    std::stringstream ss;
    ss << build_source_branch()
       << build_muxer_config()
       << build_visualization_branch()
       << build_inference_branches();
    return ss.str();
}

GstElement* PipelineBuilder::create_pipeline(GError** error) {
    std::string pipeline_desc = build_pipeline_description();
    return gst_parse_launch(pipeline_desc.c_str(), error);
}

// PipelineProbeAttacher implementation
bool PipelineProbeAttacher::attach_probe_to_element(
    GstElement* pipeline,
    const char* element_name,
    GstPadProbeCallback callback,
    gpointer user_data,
    GDestroyNotify destroy_notify,
    bool attach_to_src)
{
    GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), element_name);
    if (!element) {
        std::cerr << "Could not find element named '" << element_name << "'\n";
        return false;
    }

    const char* pad_name = attach_to_src ? "src" : "sink";
    GstPad* pad = gst_element_get_static_pad(element, pad_name);
    if (!pad) {
        std::cerr << "Could not get " << pad_name << " pad of '" << element_name << "'\n";
        gst_object_unref(element);
        return false;
    }

    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, callback, user_data, destroy_notify);
    gst_object_unref(pad);
    gst_object_unref(element);

    return true;
}
