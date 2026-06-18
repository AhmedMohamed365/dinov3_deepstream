#pragma once

#include <map>
#include <opencv2/opencv.hpp>
#include "utils/gst_headers.h"

// Utility to convert a binary mask for a given label into convex‑hull bounding boxes
// and attach them as NvDsObjectMeta to the frame meta. The tracker will then
// interpolate these objects across skipped frames, allowing the original mask
// (or depth) values to be copied forward.
static void add_segmentation_objects(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const cv::Mat &mask, int label_id)
{
    // Find contours of the mask (non‑zero pixels)
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat binary;
    mask.convertTo(binary, CV_8U);
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &cnt : contours) {
        if (cnt.empty()) continue;
        // Compute convex hull and then bounding rectangle
        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);
        cv::Rect bbox = cv::boundingRect(hull);

        // Create NvDsObjectMeta and fill fields
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj_meta) continue;
        obj_meta->class_id = label_id;
        obj_meta->object_id = -1; // tracker will assign unique IDs
        obj_meta->confidence = 1.0; // full confidence for segmentation objects
        obj_meta->rect_params.left = static_cast<float>(bbox.x);
        obj_meta->rect_params.top = static_cast<float>(bbox.y);
        obj_meta->rect_params.width = static_cast<float>(bbox.width);
        obj_meta->rect_params.height = static_cast<float>(bbox.height);
        obj_meta->rect_params.border_width = 2;
        obj_meta->rect_params.border_color.red = 0.0;
        obj_meta->rect_params.border_color.green = 1.0;
        obj_meta->rect_params.border_color.blue = 0.0;
        obj_meta->rect_params.border_color.alpha = 1.0;
        // Attach to the frame meta list
        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
    }
}

// Iterate over all label masks (key = label id, value = binary mask) and generate objects
static void generate_segmentation_objects(NvDsBatchMeta *batch_meta, const std::map<int, cv::Mat> &label_masks)
{
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;
        for (const auto &pair : label_masks) {
            int label_id = pair.first;
            const cv::Mat &mask = pair.second;
            if (mask.empty()) continue;
            add_segmentation_objects(batch_meta, frame_meta, mask, label_id);
        }
    }
}
