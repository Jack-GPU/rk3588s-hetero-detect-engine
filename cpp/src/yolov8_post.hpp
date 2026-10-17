// YOLOv8 单输出后处理：解码 + 按类 NMS（与 python/board_infer_yolov8.py 对齐）
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct YoloDet {
  int cls = 0;
  float score = 0.f;
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct LetterboxMeta {
  float scale = 1.f;
  int pad_x = 0;
  int pad_y = 0;
  int orig_w = 0;
  int orig_h = 0;
};

inline LetterboxMeta make_letterbox_meta(int src_w, int src_h, int model_w,
                                        int model_h) {
  LetterboxMeta m;
  m.orig_w = src_w;
  m.orig_h = src_h;
  m.scale = std::min(model_w / (float)src_w, model_h / (float)src_h);
  int nw = (int)std::round(src_w * m.scale);
  int nh = (int)std::round(src_h * m.scale);
  m.pad_x = (model_w - nw) / 2;
  m.pad_y = (model_h - nh) / 2;
  return m;
}

inline float iou_box(float ax1, float ay1, float ax2, float ay2, float bx1,
                     float by1, float bx2, float by2) {
  float xx1 = std::max(ax1, bx1);
  float yy1 = std::max(ay1, by1);
  float xx2 = std::min(ax2, bx2);
  float yy2 = std::min(ay2, by2);
  float w = std::max(0.f, xx2 - xx1);
  float h = std::max(0.f, yy2 - yy1);
  float inter = w * h;
  float a = std::max(0.f, ax2 - ax1) * std::max(0.f, ay2 - ay1);
  float b = std::max(0.f, bx2 - bx1) * std::max(0.f, by2 - by1);
  return inter / (a + b - inter + 1e-6f);
}

// out: (84, 8400) row-major；按「类别平面」顺序扫，利用连续访存
inline std::vector<YoloDet> yolov8_postprocess(const float* out, int num_anchors,
                                               int num_logits,
                                               const LetterboxMeta& lb,
                                               float conf_thres = 0.25f,
                                               float iou_thres = 0.45f,
                                               int max_det = 100) {
  std::vector<YoloDet> cands;
  cands.reserve(512);
  const int nc = num_logits - 4;
  if (nc <= 0 || !out) return cands;

  thread_local std::vector<float> best_s;
  thread_local std::vector<int> best_c;
  best_s.assign(num_anchors, -1.f);
  best_c.assign(num_anchors, 0);
  for (int c = 0; c < nc; ++c) {
    const float* scores = out + (4 + c) * num_anchors;
    for (int a = 0; a < num_anchors; ++a) {
      float s = scores[a];
      if (s > best_s[a]) {
        best_s[a] = s;
        best_c[a] = c;
      }
    }
  }

  const float* px = out + 0 * num_anchors;
  const float* py = out + 1 * num_anchors;
  const float* pw = out + 2 * num_anchors;
  const float* ph = out + 3 * num_anchors;
  for (int a = 0; a < num_anchors; ++a) {
    if (best_s[a] < conf_thres) continue;
    float cx = px[a], cy = py[a], bw = pw[a], bh = ph[a];
    float x1 = (cx - bw * 0.5f - lb.pad_x) / lb.scale;
    float y1 = (cy - bh * 0.5f - lb.pad_y) / lb.scale;
    float x2 = (cx + bw * 0.5f - lb.pad_x) / lb.scale;
    float y2 = (cy + bh * 0.5f - lb.pad_y) / lb.scale;
    x1 = std::max(0.f, std::min(x1, (float)lb.orig_w - 1.f));
    x2 = std::max(0.f, std::min(x2, (float)lb.orig_w - 1.f));
    y1 = std::max(0.f, std::min(y1, (float)lb.orig_h - 1.f));
    y2 = std::max(0.f, std::min(y2, (float)lb.orig_h - 1.f));
    cands.push_back({best_c[a], best_s[a], x1, y1, x2, y2});
  }

  std::vector<YoloDet> dets;
  dets.reserve(std::min(max_det, (int)cands.size()));
  // 按类桶，避免对每个 class 全表扫描
  std::vector<std::vector<int>> buckets(nc);
  for (int i = 0; i < (int)cands.size(); ++i)
    buckets[cands[i].cls].push_back(i);

  for (int c = 0; c < nc; ++c) {
    auto& idx = buckets[c];
    if (idx.empty()) continue;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return cands[a].score > cands[b].score; });
    std::vector<char> killed(idx.size(), 0);
    for (size_t i = 0; i < idx.size(); ++i) {
      if (killed[i]) continue;
      const auto& bi = cands[idx[i]];
      dets.push_back(bi);
      if ((int)dets.size() >= max_det) break;
      for (size_t j = i + 1; j < idx.size(); ++j) {
        if (killed[j]) continue;
        const auto& bj = cands[idx[j]];
        if (iou_box(bi.x1, bi.y1, bi.x2, bi.y2, bj.x1, bj.y1, bj.x2, bj.y2) >
            iou_thres)
          killed[j] = 1;
      }
    }
    if ((int)dets.size() >= max_det) break;
  }
  std::sort(dets.begin(), dets.end(),
            [](const YoloDet& a, const YoloDet& b) { return a.score > b.score; });
  if ((int)dets.size() > max_det) dets.resize(max_det);
  return dets;
}
