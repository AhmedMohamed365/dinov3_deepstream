#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"
#include <string>

// Pipeline builder for GStreamer pipeline construction
class PipelineBuilder {
public:
    PipelineBuilder(const AppConfig& cfg) : config(cfg) {}

    // Build complete pipeline description string
    std::string build_pipeline_description();

    // Create GStreamer pipeline from description
    GstElement* create_pipeline(GError** error);

private:
    const AppConfig& config;

    // Helper methods for building pipeline sections
    std::string build_source_branch();
    std::string build_muxer_config();
    std::string build_visualization_branch();
    std::string build_inference_branches();
    std::string build_tiled_muxer();  // Helper for tiled mode muxer setup
};

// Helper class for attaching probes to pipeline elements
class PipelineProbeAttacher {
public:
    // Attach probe to a named element's src or sink pad
    static bool attach_probe_to_element(
        GstElement* pipeline,
        const char* element_name,
        GstPadProbeCallback callback,
        gpointer user_data,
        GDestroyNotify destroy_notify,
        bool attach_to_src = true);  // true = src pad, false = sink pad
};
