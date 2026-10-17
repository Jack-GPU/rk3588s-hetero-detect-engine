// 预处理对照 A/B（服务时间，不含排队）
// A: CPU letterbox + memcpy → RKNN input mem
// B: RGA letterbox 写入 rknn_create_mem 的 dma-buf fd（零拷贝导入）
//
// 用法（板上）：
//   ./preprocess_ab_bench model.rknn frame_1080p.rgb 1920 1080 [loops]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "im2d.hpp"
#include "rga.h"
#include "rknn_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static double percentile(std::vector<double>& v, double p) {
  if (v.empty()) return nan("");
  std::sort(v.begin(), v.end());
  double k = (v.size() - 1) * (p / 100.0);
  size_t f = (size_t)std::floor(k);
  size_t c = (size_t)std::ceil(k);
  if (f == c) return v[f];
  return v[f] * (c - k) + v[c] * (k - f);
}

static void cpu_letterbox_rgb(const uint8_t* src, int sw, int sh,
                              uint8_t* dst, int dw, int dh, uint8_t pad) {
  float scale = std::min(dw / (float)sw, dh / (float)sh);
  int nw = (int)std::round(sw * scale);
  int nh = (int)std::round(sh * scale);
  int left = (dw - nw) / 2;
  int top = (dh - nh) / 2;
  std::memset(dst, pad, (size_t)dw * dh * 3);

  for (int y = 0; y < nh; ++y) {
    float sy = (y + 0.5f) / scale - 0.5f;
    int y0 = std::max(0, std::min(sh - 1, (int)std::floor(sy)));
    int y1 = std::max(0, std::min(sh - 1, y0 + 1));
    float fy = sy - y0;
    for (int x = 0; x < nw; ++x) {
      float sx = (x + 0.5f) / scale - 0.5f;
      int x0 = std::max(0, std::min(sw - 1, (int)std::floor(sx)));
      int x1 = std::max(0, std::min(sw - 1, x0 + 1));
      float fx = sx - x0;
      for (int c = 0; c < 3; ++c) {
        float v00 = src[(y0 * sw + x0) * 3 + c];
        float v01 = src[(y0 * sw + x1) * 3 + c];
        float v10 = src[(y1 * sw + x0) * 3 + c];
        float v11 = src[(y1 * sw + x1) * 3 + c];
        float v0 = v00 + (v01 - v00) * fx;
        float v1 = v10 + (v11 - v10) * fx;
        dst[((top + y) * dw + (left + x)) * 3 + c] =
            (uint8_t)std::round(v0 + (v1 - v0) * fy);
      }
    }
  }
}

static int letterbox_geometry(int sw, int sh, int dw, int dh, int* nw, int* nh,
                              int* left, int* top) {
  float scale = std::min(dw / (float)sw, dh / (float)sh);
  *nw = (int)std::round(sw * scale);
  *nh = (int)std::round(sh * scale);
  *left = (dw - *nw) / 2;
  *top = (dh - *nh) / 2;
  return 0;
}

static const char* rknn_err(int code) {
  static char buf[64];
  std::snprintf(buf, sizeof(buf), "rknn_err=%d", code);
  return buf;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s model.rknn [image.jpg|frame.rgb] [w h] [loops]\n",
                 argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* image_path = argc > 2 ? argv[2] : "bus.jpg";
  int loops = 200;
  int sw = 0, sh = 0, sc = 0;
  uint8_t* src = nullptr;
  bool free_src = false;

  std::string path(image_path);
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".rgb" && argc >= 5) {
    sw = std::atoi(argv[3]);
    sh = std::atoi(argv[4]);
    if (argc >= 6) loops = std::atoi(argv[5]);
    size_t nbytes = (size_t)sw * sh * 3;
    src = (uint8_t*)std::malloc(nbytes);
    free_src = true;
    std::ifstream ifs(image_path, std::ios::binary);
    if (!ifs.read((char*)src, nbytes)) {
      std::fprintf(stderr, "read rgb failed\n");
      return 1;
    }
  } else {
    if (argc >= 4 && path.substr(path.size() - 4) != ".rgb")
      loops = std::atoi(argv[3]);
    src = stbi_load(image_path, &sw, &sh, &sc, 3);
    free_src = true;
    if (!src) {
      std::fprintf(stderr, "stbi_load failed: %s\n", image_path);
      return 1;
    }
    sc = 3;
  }

  std::printf("input %dx%d loops=%d\n", sw, sh, loops);

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, (void*)model_path, 0, 0, nullptr);
  if (ret != 0) {
    std::fprintf(stderr, "rknn_init failed %s\n", rknn_err(ret));
    return 1;
  }

  rknn_input_output_num io_num{};
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != 0) return 1;

  rknn_tensor_attr in_attr{};
  in_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  if (ret != 0) return 1;

  // 期望 NHWC uint8 640x640x3
  int model_h = 0, model_w = 0, model_c = 0;
  if (in_attr.fmt == RKNN_TENSOR_NHWC) {
    model_h = (int)in_attr.dims[1];
    model_w = (int)in_attr.dims[2];
    model_c = (int)in_attr.dims[3];
  } else {
    model_c = (int)in_attr.dims[1];
    model_h = (int)in_attr.dims[2];
    model_w = (int)in_attr.dims[3];
  }
  std::printf("model input %dx%dx%d fmt=%d type=%d size_with_stride=%u fd_ready\n",
              model_w, model_h, model_c, (int)in_attr.fmt, (int)in_attr.type,
              in_attr.size_with_stride);

  in_attr.type = RKNN_TENSOR_UINT8;
  in_attr.fmt = RKNN_TENSOR_NHWC;

  rknn_tensor_mem* input_mem =
      rknn_create_mem(ctx, in_attr.size_with_stride);
  if (!input_mem) {
    std::fprintf(stderr, "rknn_create_mem failed\n");
    return 1;
  }
  ret = rknn_set_io_mem(ctx, input_mem, &in_attr);
  if (ret != 0) {
    std::fprintf(stderr, "rknn_set_io_mem failed %s\n", rknn_err(ret));
    return 1;
  }
  std::printf("input_mem fd=%d virt=%p size=%u\n", input_mem->fd,
              input_mem->virt_addr, input_mem->size);

  // output mems as float for quick sanity (optional run)
  std::vector<rknn_tensor_mem*> out_mems(io_num.n_output);
  std::vector<rknn_tensor_attr> out_attrs(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    out_attrs[i].index = i;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(out_attrs[i]));
    out_attrs[i].type = RKNN_TENSOR_FLOAT32;
    out_mems[i] =
        rknn_create_mem(ctx, out_attrs[i].n_elems * sizeof(float));
    rknn_set_io_mem(ctx, out_mems[i], &out_attrs[i]);
  }

  const int dw = model_w;
  const int dh = model_h;
  int nw = 0, nh = 0, left = 0, top = 0;
  letterbox_geometry(sw, sh, dw, dh, &nw, &nh, &left, &top);
  std::printf("letterbox nw=%d nh=%d left=%d top=%d\n", nw, nh, left, top);

  std::vector<uint8_t> cpu_buf((size_t)dw * dh * 3);
  int wstride = dw;  // 640 already 16-aligned
  int hstride = dh;

  // warmup
  for (int i = 0; i < 10; ++i) {
    cpu_letterbox_rgb(src, sw, sh, cpu_buf.data(), dw, dh, 114);
    std::memcpy(input_mem->virt_addr, cpu_buf.data(), cpu_buf.size());
    rknn_run(ctx, nullptr);
  }

  std::vector<double> a_pre, a_infer, b_pre, b_infer;

  using clock = std::chrono::steady_clock;

  // -------- Path A --------
  for (int i = 0; i < loops; ++i) {
    auto t0 = clock::now();
    cpu_letterbox_rgb(src, sw, sh, cpu_buf.data(), dw, dh, 114);
    std::memcpy(input_mem->virt_addr, cpu_buf.data(), cpu_buf.size());
    auto t1 = clock::now();
    ret = rknn_run(ctx, nullptr);
    auto t2 = clock::now();
    if (ret != 0) {
      std::fprintf(stderr, "A rknn_run failed %d\n", ret);
      return 1;
    }
    a_pre.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    a_infer.push_back(
        std::chrono::duration<double, std::milli>(t2 - t1).count());
  }

  // -------- Path B --------
  rga_buffer_t src_buf =
      wrapbuffer_virtualaddr((void*)src, sw, sh, RK_FORMAT_RGB_888);
  rga_buffer_t dst_buf =
      wrapbuffer_fd(input_mem->fd, dw, dh, RK_FORMAT_RGB_888, wstride, hstride);

  im_rect src_rect{};
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.width = sw;
  src_rect.height = sh;
  im_rect dst_rect{};
  dst_rect.x = left;
  dst_rect.y = top;
  dst_rect.width = nw;
  dst_rect.height = nh;
  rga_buffer_t pat{};
  im_rect pat_rect{};

  // warmup RGA
  for (int i = 0; i < 10; ++i) {
    std::memset(input_mem->virt_addr, 114, input_mem->size);
    IM_STATUS st = improcess(src_buf, dst_buf, pat, src_rect, dst_rect, pat_rect,
                             IM_SYNC);
    if (st != IM_STATUS_SUCCESS) {
      std::fprintf(stderr, "RGA improcess failed: %s\n", imStrError(st));
      return 1;
    }
    rknn_run(ctx, nullptr);
  }

  for (int i = 0; i < loops; ++i) {
    auto t0 = clock::now();
    // pad color 114；memset 填充 letterbox 边，再 RGA 写有效区
    std::memset(input_mem->virt_addr, 114, (size_t)dw * dh * 3);
    IM_STATUS st = improcess(src_buf, dst_buf, pat, src_rect, dst_rect, pat_rect,
                             IM_SYNC);
    auto t1 = clock::now();
    if (st != IM_STATUS_SUCCESS) {
      std::fprintf(stderr, "RGA improcess failed: %s\n", imStrError(st));
      return 1;
    }
    ret = rknn_run(ctx, nullptr);
    auto t2 = clock::now();
    if (ret != 0) {
      std::fprintf(stderr, "B rknn_run failed %d\n", ret);
      return 1;
    }
    b_pre.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    b_infer.push_back(
        std::chrono::duration<double, std::milli>(t2 - t1).count());
  }

  // sanity: class score max from last run
  float* out0 = (float*)out_mems[0]->virt_addr;
  size_t n = out_attrs[0].n_elems;
  // output layout (1,84,8400) flattened — max over class channels roughly
  float cls_max = 0.f;
  if (n >= 84 * 8400) {
    for (int a = 0; a < 8400; ++a) {
      for (int c = 4; c < 84; ++c) {
        // NCHW-like [1,84,8400] → idx = c*8400 + a
        cls_max = std::max(cls_max, out0[c * 8400 + a]);
      }
    }
  }

  auto dump = [](const char* name, std::vector<double> v) {
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double p50 = percentile(v, 50);
    double p99 = percentile(v, 99);
    std::printf("  \"%s\": {\"p50\": %.3f, \"p99\": %.3f, \"mean\": %.3f},\n",
                name, p50, p99, mean);
  };

  std::printf("{\n");
  std::printf("  \"pipeline_A\": \"cpu_letterbox + memcpy -> rknn_mem\",\n");
  std::printf(
      "  \"pipeline_B\": \"memset_pad + RGA improcess -> rknn_mem.fd\",\n");
  std::printf("  \"note\": \"服务时间；rknn_set_io_mem 只在初始化调用一次\",\n");
  std::printf("  \"input\": \"%dx%d\",\n", sw, sh);
  std::printf("  \"model_input\": \"%dx%d\",\n", dw, dh);
  std::printf("  \"loops\": %d,\n", loops);
  std::printf("  \"input_mem_fd\": %d,\n", input_mem->fd);
  std::printf("  \"cls_max_last\": %.4f,\n", cls_max);
  dump("A_preprocess_ms", a_pre);
  dump("A_infer_ms", a_infer);
  dump("B_preprocess_ms", b_pre);
  dump("B_infer_ms", b_infer);
  // remove trailing comma awkwardly — print speedup
  double a50 = percentile(a_pre, 50);
  double b50 = percentile(b_pre, 50);
  std::printf("  \"speedup_preprocess_p50\": %.3f\n", a50 / b50);
  std::printf("}\n");

  for (auto* m : out_mems) rknn_destroy_mem(ctx, m);
  rknn_destroy_mem(ctx, input_mem);
  rknn_destroy(ctx);
  if (free_src) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".rgb")
      std::free(src);
    else
      stbi_image_free(src);
  }
  return 0;
}
