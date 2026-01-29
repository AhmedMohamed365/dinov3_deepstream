#include <nvdsinfer_custom_impl.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// DeepStream uses this struct for decoded boxes. :contentReference[oaicite:1]{index=1}

static inline float sigmoidf(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

static inline float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

static int getenv_int(const char* key, int fallback) {
  const char* s = std::getenv(key);
  if (!s || !*s) return fallback;
  int v = std::atoi(s);
  return (v > 0) ? v : fallback;
}

static const NvDsInferLayerInfo* find_layer(
    const std::vector<NvDsInferLayerInfo>& layers,
    const char* name)
{
  for (const auto& l : layers) {
    if (l.layerName && std::strcmp(l.layerName, name) == 0) return &l;
  }
  return nullptr;
}

// Extract (C,H,W) from dims. Expected NCHW: [1,C,H,W].
static bool get_chw(const NvDsInferDims& d, int& C, int& H, int& W) {
  if (d.numDims == 4) { // [N,C,H,W]
    C = d.d[1]; H = d.d[2]; W = d.d[3];
    return true;
  }
  if (d.numDims == 3) { // [C,H,W] (sometimes)
    C = d.d[0]; H = d.d[1]; W = d.d[2];
    return true;
  }
  return false;
}

// Main parser
extern "C" bool NvDsInferParseCustomDetection(
    const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  // ---- Find the 9 expected outputs by name ----
  const NvDsInferLayerInfo* cls1 = find_layer(outputLayersInfo, "cls1");
  const NvDsInferLayerInfo* cls2 = find_layer(outputLayersInfo, "cls2");
  const NvDsInferLayerInfo* cls3 = find_layer(outputLayersInfo, "cls3");

  const NvDsInferLayerInfo* reg1 = find_layer(outputLayersInfo, "reg1");
  const NvDsInferLayerInfo* reg2 = find_layer(outputLayersInfo, "reg2");
  const NvDsInferLayerInfo* reg3 = find_layer(outputLayersInfo, "reg3");

  const NvDsInferLayerInfo* ctr1 = find_layer(outputLayersInfo, "ctr1");
  const NvDsInferLayerInfo* ctr2 = find_layer(outputLayersInfo, "ctr2");
  const NvDsInferLayerInfo* ctr3 = find_layer(outputLayersInfo, "ctr3");

  if (!cls1 || !cls2 || !cls3 || !reg1 || !reg2 || !reg3 || !ctr1 || !ctr2 || !ctr3) {
    // Layer names must match what TensorRT reports.
    return false;
  }

  // Pointers are host pointers for parsing (nvinfer runs parsing on CPU).
  const float* cls_ptrs[3] = {
    static_cast<const float*>(cls1->buffer),
    static_cast<const float*>(cls2->buffer),
    static_cast<const float*>(cls3->buffer)
  };
  const float* reg_ptrs[3] = {
    static_cast<const float*>(reg1->buffer),
    static_cast<const float*>(reg2->buffer),
    static_cast<const float*>(reg3->buffer)
  };
  const float* ctr_ptrs[3] = {
    static_cast<const float*>(ctr1->buffer),
    static_cast<const float*>(ctr2->buffer),
    static_cast<const float*>(ctr3->buffer)
  };

  if (!cls_ptrs[0] || !reg_ptrs[0] || !ctr_ptrs[0]) return false;

  // Read dims per level from cls (should match reg/ctr spatially)
  int C = 0;
  int Hs[3] = {0,0,0};
  int Ws[3] = {0,0,0};

  for (int l = 0; l < 3; ++l) {
    int c=0,h=0,w=0;
    const NvDsInferLayerInfo* cl = (l==0?cls1:(l==1?cls2:cls3));
    if (!get_chw(cl->inferDims, c, h, w)) return false;
    if (l == 0) C = c;
    if (c != C) return false;
    Hs[l] = h; Ws[l] = w;
  }

  // Image size used for decoding centers
  // networkInfo.width/height may not reflect original image resolution
  // when the head consumes features instead of images.
  // Override with env vars:
  //   FCOS_IMG_W=800 FCOS_IMG_H=800
  //
  // If networkInfo seems too small, we default to 800.
  int imgW = getenv_int("FCOS_IMG_W", networkInfo.width);
  int imgH = getenv_int("FCOS_IMG_H", networkInfo.height);
  if (imgW < 64) imgW = 640;
  if (imgH < 64) imgH = 640;

  // Thresholds: DeepStream supplies per-class precluster thresholds here. :contentReference[oaicite:2]{index=2}
  const int numClasses = std::min((int)detectionParams.numClassesConfigured, C);

  // Decode per level
  for (int lvl = 0; lvl < 3; ++lvl) {
    const int H = Hs[lvl], W = Ws[lvl];
    const int HW = H * W;

    const float strideX = (float)imgW / (float)W;
    const float strideY = (float)imgH / (float)H;

    const float* cls = cls_ptrs[lvl]; // [C,H,W]
    const float* reg = reg_ptrs[lvl]; // [4,H,W]  l,t,r,b in pixels
    const float* ctr = ctr_ptrs[lvl]; // [1,H,W]

    // For each location, pick best class (reduces 80x explosion)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int idx = y * W + x;

        const float ctr_p = sigmoidf(ctr[idx]);

        int bestC = -1;
        float bestScore = 0.0f;

        // scan classes at this location
        for (int c = 0; c < numClasses; ++c) {
          const float p = sigmoidf(cls[c * HW + idx]) * ctr_p;
          if (p > bestScore) {
            bestScore = p;
            bestC = c;
          }
        }
        if (bestC < 0) continue;

        const float thr = detectionParams.perClassPreclusterThreshold[bestC];
        if (bestScore < thr) continue;

        // decode bbox
        const float l = reg[0 * HW + idx];
        const float t = reg[1 * HW + idx];
        const float r = reg[2 * HW + idx];
        const float b = reg[3 * HW + idx];

        const float xc = ((float)x + 0.5f) * strideX;
        const float yc = ((float)y + 0.5f) * strideY;

        float x1 = xc - l;
        float y1 = yc - t;
        float x2 = xc + r;
        float y2 = yc + b;

        x1 = clampf(x1, 0.0f, (float)(imgW - 1));
        y1 = clampf(y1, 0.0f, (float)(imgH - 1));
        x2 = clampf(x2, 0.0f, (float)(imgW - 1));
        y2 = clampf(y2, 0.0f, (float)(imgH - 1));

        float bw = x2 - x1;
        float bh = y2 - y1;
        if (bw <= 1.0f || bh <= 1.0f) continue;

        NvDsInferParseObjectInfo obj;
        obj.left = x1;
        obj.top = y1;
        obj.width = bw;
        obj.height = bh;
        obj.detectionConfidence = bestScore;
        obj.classId = bestC;

        objectList.push_back(obj);
      }
    }
  }

  return true;
}

// Verify function signature
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDetection);