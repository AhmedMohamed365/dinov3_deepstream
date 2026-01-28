#pragma once

#include "gst_headers.h"

#include <cuda_runtime_api.h>

#include <cuda_fp16.h>

#include <bits/stdc++.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <unordered_set>

static inline size_t elem_size_from_infer_dtype(NvDsInferDataType t) {
    switch (t) {
      case NvDsInferDataType::FLOAT: return 4;
      case NvDsInferDataType::HALF:  return 2;
      case NvDsInferDataType::INT8:  return 1;
      case NvDsInferDataType::INT32: return 4;
  #ifdef NvDsInferDataType::BOOL
      case NvDsInferDataType::BOOL:  return 1;
  #endif
      default: return 0;
    }
  }
  
  static inline size_t volume_from_dims(const NvDsInferDims& d) {
    size_t v = 1;
    for (int i = 0; i < d.numDims; ++i) v *= (size_t)d.d[i];
    return v;
  }

  // Check if buffer color format is part of NV12 family
  // Supports standard, extended range, and different color spaces (BT.601, BT.709, BT.2020)
  static inline bool is_nv12_color_format(NvBufSurfaceColorFormat fmt) {
    return fmt == NVBUF_COLOR_FORMAT_NV12 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_2020;
  }

  // DeepStream calls copy_func with (data = NvDsUserMeta*).
  static gpointer preprocess_batchmeta_copy_func(gpointer data, gpointer user_data) {
    (void)user_data;
    NvDsUserMeta *src_umeta = (NvDsUserMeta *)data;
    auto *src = (GstNvDsPreProcessBatchMeta *)src_umeta->user_meta_data;
    if (!src) return nullptr;
  
    auto *dst = new GstNvDsPreProcessBatchMeta();
    dst->target_unique_ids = src->target_unique_ids;
    dst->roi_vector        = src->roi_vector;
    dst->private_data      = nullptr;
  
    if (src->tensor_meta) {
      dst->tensor_meta = new NvDsPreProcessTensorMeta();
      *(dst->tensor_meta) = *(src->tensor_meta);   // std::string + std::vector copy OK
      // NOTE: raw_tensor_buffer pointer is copied as-is (we want the same GPU pointer).
    } else {
      dst->tensor_meta = nullptr;
    }
    return (gpointer)dst;
  }
  
  // DeepStream calls release_func with (data = NvDsUserMeta*).
  static void preprocess_batchmeta_release_func(gpointer data, gpointer user_data) {
    (void)user_data;
    NvDsUserMeta *umeta = (NvDsUserMeta *)data;
    auto *meta = (GstNvDsPreProcessBatchMeta *)umeta->user_meta_data;
  
    if (meta) {
      if (meta->tensor_meta) {
        delete meta->tensor_meta;
        meta->tensor_meta = nullptr;
      }
      delete meta;
      umeta->user_meta_data = nullptr;
    }
  }
  
  // avoid attaching twice per buffer
  static bool already_has_preprocess_for_uids(
    NvDsBatchMeta* batch_meta,
    std::initializer_list<guint64> required_uids)
{
  if (!batch_meta) return false;

  // Put required into a set for easy erase/check
  std::unordered_set<guint64> req(required_uids.begin(), required_uids.end());
    if (req.empty()) return false;

    for (NvDsMetaList* l = batch_meta->batch_user_meta_list; l != nullptr; l = l->next) {
      NvDsUserMeta* um = (NvDsUserMeta*)l->data;
      if (!um) continue;
      if (um->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META) continue;

      auto* pbm = (GstNvDsPreProcessBatchMeta*)um->user_meta_data;
      if (!pbm) continue;

      // If this meta covers all required IDs, we're done
      size_t matched = 0;
      for (auto uid : pbm->target_unique_ids) {
        if (req.find(uid) != req.end()) matched++;
      }

      if (matched == req.size()) return true;
    }

    return false;
  }



  static GstNvDsPreProcessBatchMeta*
find_preprocess_meta_for_uid(NvDsBatchMeta* batch_meta, guint64 target_uid)
{
  for (NvDsMetaList* l = batch_meta->batch_user_meta_list; l; l = l->next) {
    NvDsUserMeta* um = (NvDsUserMeta*) l->data;
    if (!um) continue;
    if (um->base_meta.meta_type != (NvDsMetaType)NVDS_PREPROCESS_BATCH_META) continue;

    auto* pbm = (GstNvDsPreProcessBatchMeta*) um->user_meta_data;
    if (!pbm) continue;

    for (auto uid : pbm->target_unique_ids) {
      if (uid == target_uid) return pbm;
    }
  }
  return nullptr;
}

static NvDsInferTensorMeta*
find_tensor_meta_in_user_meta_list(NvDsMetaList* user_meta_list, guint uid)
{
  for (NvDsMetaList* l = user_meta_list; l; l = l->next) {
    NvDsUserMeta* um = (NvDsUserMeta*) l->data;
    if (!um) continue;
    if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;

    auto* tmeta = (NvDsInferTensorMeta*) um->user_meta_data;
    if (tmeta && tmeta->unique_id == uid) return tmeta;
  }
  return nullptr;
}

static int find_layer_index(NvDsInferTensorMeta* tmeta, const std::string& name) {
  for (unsigned i = 0; i < tmeta->num_output_layers; ++i) {
    const char* ln = tmeta->output_layers_info[i].layerName;
    if (ln && name == ln) return (int)i;
  }
  return -1;
}

// SEGMENTATION UTILS
static gpointer segmeta_copy_func(gpointer data, gpointer user_data)
{
  (void)user_data;

  NvDsUserMeta* src_umeta = (NvDsUserMeta*)data;
  if (!src_umeta) return nullptr;

  auto* src = (NvDsInferSegmentationMeta*)src_umeta->user_meta_data;
  if (!src) return nullptr;

  auto* dst = (NvDsInferSegmentationMeta*)g_malloc0(sizeof(NvDsInferSegmentationMeta));
  dst->unique_id = src->unique_id;
  dst->classes   = src->classes;
  dst->width     = src->width;
  dst->height    = src->height;

  // Deep copy class_map
  if (src->class_map && src->width > 0 && src->height > 0) {
    gsize bytes = (gsize)src->width * (gsize)src->height * sizeof(int);
    dst->class_map = (int*)g_memdup2(src->class_map, bytes);
  } else {
    dst->class_map = nullptr;
  }

  // Only set this if YOU allocate it; else keep null.
  dst->class_probabilities_map = nullptr;

  return (gpointer)dst; // <- DeepStream will store this as dest_umeta->user_meta_data
}

static void segmeta_release_func(gpointer data, gpointer user_data)
{
  (void)user_data;

  NvDsUserMeta* umeta = (NvDsUserMeta*)data;
  if (!umeta) return;

  auto* m = (NvDsInferSegmentationMeta*)umeta->user_meta_data;
  if (!m) return;

  if (m->class_map) g_free(m->class_map);
  m->class_map = nullptr;

  // Only free if you allocated it (recommended: keep it null always)
  if (m->class_probabilities_map) g_free(m->class_probabilities_map);
  m->class_probabilities_map = nullptr;

  g_free(m);
  umeta->user_meta_data = nullptr;
}


static bool frame_has_segmeta(NvDsFrameMeta* fmeta) {
  for (NvDsMetaList* l = fmeta->frame_user_meta_list; l; l = l->next) {
    auto* um = (NvDsUserMeta*)l->data;
    if (!um) continue;
    if (um->base_meta.meta_type == NVDSINFER_SEGMENTATION_META) return true;
  }
  return false;
}


static inline const char* class_name(int id) {
  // TODO: replace with your real 134-class names
  static thread_local char buf[32];
  if (id == 1) return "person";
  std::snprintf(buf, sizeof(buf), "class_%d", id);
  return buf;
}

/**
 * Copy NvBufSurface to ensure independence when buffers are shared via tee.
 *
 * The tee element shares the same NvBufSurface pointer across branches.
 * This function creates an independent GPU memory copy and replaces the
 * buffer's surface pointer.
 *
 * @param buf GstBuffer containing the surface
 * @param in_out_map GstMapInfo with currently mapped surface (will be updated)
 * @param surface Current NvBufSurface pointer
 * @return New NvBufSurface pointer, or nullptr on failure
 */
// Copies buffer surface only when needed for branch independence
// Checks buffer memory to detect if surface is shared across branches
// Single branch: surface not shared, no copy needed (fixes segfault)
// Multiple branches: surface shared via tee, must copy (prevents mixed frames)
static inline NvBufSurface* copy_and_replace_buffer_surface(
    GstBuffer* buf,
    GstMapInfo& in_out_map,
    NvBufSurface* surface)
{
    if (!buf || !surface) return nullptr;

    // Check if buffer memory is actually shared (tee shares memory across branches)
    // If memory is writable, it's not shared - single branch, no copy needed
    if (gst_buffer_n_memory(buf) == 1) {
        GstMemory* mem = gst_buffer_peek_memory(buf, 0);
        if (mem && gst_memory_is_writable(mem)) {
            // Memory is exclusive to this branch - no copy needed
            return surface;
        }
    }

    // Memory is shared across branches - need independent copy
    // Create new surface with identical parameters
    NvBufSurface* new_surface = nullptr;
    NvBufSurfaceCreateParams create_params{};
    create_params.gpuId = surface->gpuId;
    create_params.width = surface->surfaceList[0].width;
    create_params.height = surface->surfaceList[0].height;
    create_params.size = 0;  // Auto-calculate
    create_params.colorFormat = surface->surfaceList[0].colorFormat;
    create_params.layout = surface->surfaceList[0].layout;
    create_params.memType = surface->memType;

    if (NvBufSurfaceCreate(&new_surface, surface->batchSize, &create_params) != 0) {
        return nullptr;
    }

    // Copy surface contents (GPU-to-GPU memcpy)
    if (NvBufSurfaceCopy(surface, new_surface) != 0) {
        NvBufSurfaceDestroy(new_surface);
        return nullptr;
    }

    // Unmap old surface
    gst_buffer_unmap(buf, &in_out_map);

    // Buffer is shared - remove old memory and replace with copy
    gst_buffer_remove_all_memory(buf);

    // Wrap new surface - REMOVED READONLY to allow downstream writes
    GstMemory* mem = gst_memory_new_wrapped(
        GST_MEMORY_FLAG_NO_SHARE,  // Keep NO_SHARE, removed READONLY
        new_surface,
        sizeof(NvBufSurface),
        0,
        sizeof(NvBufSurface),
        new_surface,
        [](gpointer data) { NvBufSurfaceDestroy((NvBufSurface*)data); });

    gst_buffer_append_memory(buf, mem);

    // Re-map with READWRITE to allow downstream modifications
    if (!gst_buffer_map(buf, &in_out_map, GST_MAP_READWRITE)) {
        return nullptr;
    }

    return (NvBufSurface*)in_out_map.data;
}