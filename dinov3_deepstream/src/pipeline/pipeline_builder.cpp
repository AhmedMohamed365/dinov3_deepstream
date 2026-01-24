#include "pipeline_builder.h"
#include <iostream>
#include <sstream>

std::string PipelineBuilder::build_source_branch() {
    std::stringstream ss;
    ss << "v4l2src device=" << config.pipeline.device << " ! "
       << "video/x-raw,framerate=" << config.pipeline.framerate << "/1 ! "
       << "videoconvert ! "
       << "video/x-raw,format=RGBA ! "
       << "nvvideoconvert ! "
       << "video/x-raw(memory:NVMM),format=NV12 ! "
       << "queue ! mux.sink_0 ";
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

std::string PipelineBuilder::build_visualization_branch() {
    return "tee name=t0 "
           "t0. ! queue ! nvvideoconvert ! nveglglessink sync=false ";
}

std::string PipelineBuilder::build_inference_branches() {
    std::stringstream ss;

    // Shared DINOv3 backbone (runs once)
    ss << "t0. ! queue ! "
       << "nvinfer name=dinov3 config-file-path=" << config.model_paths.dinov3_config
       << " ! tee name=t1 allow-not-linked=true ";

    // Each branch - tee will NOT pass through buffers, forcing independent copies
    // The probes modify surfaces in-place, so we need truly independent memory

    // Depth branch
    if (config.inference_enable.depth) {
        ss << "t1. ! queue name=q_depth ! "
           << "nvinfer name=depth config-file-path="
           << config.model_paths.depth_config << " ! "
           << "nvvideoconvert name=postdepthconv ! "
           << "video/x-raw(memory:NVMM),format=NV12 ! "
           << "nveglglessink sync=false ";
    }

    // Detection branch
    if (config.inference_enable.detection) {
        ss << "t1. ! queue name=q_det ! "
           << "nvinfer name=detection config-file-path="
           << config.model_paths.detection_config << " ! "
           << "nvvideoconvert name=postdetectionconv ! "
           << "video/x-raw(memory:NVMM),format=RGBA ! nvdsosd ! "
           << "nveglglessink sync=false ";
    }

    // Segmentation branch
    if (config.inference_enable.segmentation) {
        ss << "t1. ! queue name=q_seg ! "
           << "nvinfer name=seg config-file-path="
           << config.model_paths.segmentation_config << " ! "
           << "nvvideoconvert ! "
           << "video/x-raw(memory:NVMM),format=RGBA,width=" << config.pipeline.width
           << ",height=" << config.pipeline.height << " ! "
           << "nvdsosd ! "
           << "nveglglessink sync=false ";
    }

    // Optical flow branch
    if (config.inference_enable.optical_flow) {
        ss << "t1. ! queue name=q_flow ! "
           << "nvinfer name=optical_flow config-file-path="
           << config.model_paths.optical_flow_config << " ! "
           << "nvvideoconvert name=postflowconv ! "
           << "video/x-raw(memory:NVMM),format=NV12 ! "
           << "nveglglessink sync=false";
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
