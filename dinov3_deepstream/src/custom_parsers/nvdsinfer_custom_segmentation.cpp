#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

// C-linkage to prevent C++ name mangling
extern "C" bool NvDsInferParseCustomSegmentation(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap);

static const NvDsInferLayerInfo* findLayer(
    const std::vector<NvDsInferLayerInfo>& layers,
    const std::string& name)
{
  for (auto& l : layers) {
    if (l.layerName && name == l.layerName) return &l;
  }
  return nullptr;
}

// Assumes output is CHW (C,H,W) on CPU as FLOAT (common for parsing)
// If your output comes as FP16 on host (rare), you can extend this.
extern "C" bool NvDsInferParseCustomSegmentation(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap)
{
  assert(classificationMap);

  // We do not provide a full per-class probability map (it would be huge for 134 classes).
  // nvsegvisual only needs classificationMap. Keep it null.
  classProbabilityMap = nullptr;

  const NvDsInferLayerInfo* segLayer = findLayer(outputLayersInfo, "semantic_segmentation");
  if (!segLayer) {
    // Fallback: if output-blob-names was set correctly, it is usually the first output
    if (!outputLayersInfo.empty()) segLayer = &outputLayersInfo[0];
  }
  if (!segLayer || !segLayer->buffer) {
    std::cerr << "ERROR: semantic segmentation output layer not found or buffer is null\n";
    return false;
  }

  if (segLayer->dataType != NvDsInferDataType::FLOAT) {
    std::cerr << "ERROR: expected FLOAT output for semantic_segmentation but got dtype="
              << (int)segLayer->dataType << "\n";
    return false;
  }

  NvDsInferDims d = segLayer->inferDims;

  // DeepStream usually gives output dims without batch: (C,H,W) => numDims=3
  if (d.numDims != 3) {
    std::cerr << "ERROR: expected 3 dims (C,H,W). Got numDims=" << d.numDims << "\n";
    return false;
  }

  int C = (int)d.d[0];
  int H = (int)d.d[1];
  int W = (int)d.d[2];

  printf("c=%d h=%d w=%d\n", C, H, W);
  fflush(stdout);

  if ((unsigned)C != numClasses) {
    std::cerr << "WARN: numClasses(config)=" << numClasses << " but output C=" << C
              << " (continuing with C)\n";
  }

  const float* logits = (const float*)segLayer->buffer;

  // Layout assumed: [C][H][W] contiguous => logits[c*H*W + y*W + x]
  const int HW = H * W;

  // If segmentationThreshold > 0, we treat it as a probability threshold (softmax of winner).
  const bool useProbThresh = (segmentationThreshold > 0.0f);

  fprintf(stdout,
    "seg dims C,H,W = %d,%d,%d => HW=%d | networkInfo WxH=%d,%d => %d\n",
    (int)d.d[0], (int)d.d[1], (int)d.d[2], (int)(d.d[1]*d.d[2]),
    (int)networkInfo.width, (int)networkInfo.height,
    (int)(networkInfo.width*networkInfo.height));
  fflush(stdout);

  for (int i = 0; i < HW; ++i) {
    // find argmax logit
    // printf("i = %d, total = %d\n", i, HW);
    // fflush(stdout);
    int best = 0;
    float bestv = logits[0 * HW + i];
    //printf("a i: %d\n", i);
    //fflush(stdout);

    for (int c = 1; c < C; ++c) {
      float v = logits[c * HW + i];
      if (v > bestv) { bestv = v; best = c; }
    }

    //printf("b i: %d\n", i);
    //fflush(stdout);

    if (!useProbThresh) {
      //printf("c i: %d\n", i);
      //fflush(stdout);
      classificationMap[i] = best;
      //printf("d i: %d\n", i);
      //fflush(stdout);
      continue;
    }

    printf("c i: %d\n", i);
    fflush(stdout);

    // Softmax probability of best class only:
    // p_best = 1 / sum_c exp(logit_c - bestv)
    double sumexp = 0.0;
    for (int c = 0; c < C; ++c) {
      sumexp += std::exp((double)logits[c * HW + i] - (double)bestv);
    }
    float p_best = (sumexp > 0.0) ? (float)(1.0 / sumexp) : 0.0f;

    printf("d i: %d\n", i);
    fflush(stdout);

    // If below threshold => background (0). Your background is class 0.
    classificationMap[i] = (p_best >= segmentationThreshold) ? best : 0;
    printf("e i: %d\n", i);
    fflush(stdout);
  }

  printf("Holi6\n");
  fflush(stdout);
  return true;
}

// Check prototype
CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSegmentation);