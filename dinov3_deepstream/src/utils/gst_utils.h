#pragma once

#include "gst_headers.h"

#include <cuda_runtime_api.h>

#include <cuda_fp16.h>

#include <bits/stdc++.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
  static bool already_has_preprocess_for_uid(NvDsBatchMeta *batch_meta, guint64 target_uid) {
    for (NvDsMetaList *l = batch_meta->batch_user_meta_list; l != nullptr; l = l->next) {
      NvDsUserMeta *um = (NvDsUserMeta *)l->data;
      if (!um) continue;
      if (um->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META) continue;
  
      auto *pbm = (GstNvDsPreProcessBatchMeta *)um->user_meta_data;
      if (!pbm) continue;
  
      for (auto uid : pbm->target_unique_ids) {
        if (uid == target_uid) return true;
      }
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