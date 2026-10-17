// MPP 硬解(H.264) → RGA(NV12 letterbox→RGB) → RKNN fd 零拷贝推理
// 口径：decode / rga / infer / e2e 服务时间；预热帧不计分位
//
// 板上：
//   ./mpp_rga_rknn_bench model.rknn video.h264 [max_frames]

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
#include <time.h>
#include <vector>

#include "im2d.hpp"
#include "mpp_buffer.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "rga.h"
#include "rk_mpi.h"
#include "rknn_api.h"

static double percentile(std::vector<double> v, double p) {
  if (v.empty()) return nan("");
  std::sort(v.begin(), v.end());
  double k = (v.size() - 1) * (p / 100.0);
  size_t f = (size_t)std::floor(k);
  size_t c = (size_t)std::ceil(k);
  if (f == c) return v[f];
  return v[f] * (c - k) + v[c] * (k - f);
}

static std::vector<uint8_t> read_file(const char* path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return {};
  ifs.seekg(0, std::ios::end);
  size_t n = (size_t)ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(n);
  ifs.read((char*)buf.data(), n);
  return buf;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn stream.h264 [max_frames]\n",
                 argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* h264_path = argv[2];
  int max_frames = argc > 3 ? std::atoi(argv[3]) : 120;
  const int warmup = 20;

  auto bitstream = read_file(h264_path);
  if (bitstream.empty()) {
    std::fprintf(stderr, "read h264 failed\n");
    return 1;
  }

  // ---- RKNN ----
  rknn_context rknn_ctx = 0;
  int ret = rknn_init(&rknn_ctx, (void*)model_path, 0, 0, nullptr);
  if (ret != 0) {
    std::fprintf(stderr, "rknn_init failed %d\n", ret);
    return 1;
  }
  rknn_tensor_attr in_attr{};
  in_attr.index = 0;
  rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  in_attr.type = RKNN_TENSOR_UINT8;
  in_attr.fmt = RKNN_TENSOR_NHWC;
  int model_w = (int)in_attr.dims[2];
  int model_h = (int)in_attr.dims[1];
  rknn_tensor_mem* input_mem =
      rknn_create_mem(rknn_ctx, in_attr.size_with_stride);
  rknn_set_io_mem(rknn_ctx, input_mem, &in_attr);

  rknn_input_output_num io_num{};
  rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  std::vector<rknn_tensor_mem*> out_mems(io_num.n_output);
  std::vector<rknn_tensor_attr> out_attrs(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    out_attrs[i].index = i;
    rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i],
               sizeof(out_attrs[i]));
    out_attrs[i].type = RKNN_TENSOR_FLOAT32;
    out_mems[i] =
        rknn_create_mem(rknn_ctx, out_attrs[i].n_elems * sizeof(float));
    rknn_set_io_mem(rknn_ctx, out_mems[i], &out_attrs[i]);
  }

  // ---- MPP ----
  MppCtx ctx = NULL;
  MppApi* mpi = NULL;
  ret = mpp_create(&ctx, &mpi);
  if (ret != MPP_OK) {
    std::fprintf(stderr, "mpp_create failed %d\n", ret);
    return 1;
  }
  ret = mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
  if (ret != MPP_OK) {
    std::fprintf(stderr, "mpp_init failed %d\n", ret);
    return 1;
  }
  RK_U32 need_split = 1;
  mpi->control(ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);

  MppPacket packet = NULL;
  mpp_packet_init(&packet, bitstream.data(), bitstream.size());
  mpp_packet_set_eos(packet);

  MppBufferGroup frm_grp = NULL;
  int frame_count = 0;
  int measured = 0;
  std::vector<double> decode_ms, rga_ms, infer_ms, e2e_ms;
  float cls_max = 0.f;

  using clock = std::chrono::steady_clock;

  // 整文件一次送入（annex-B）；超时退出避免空转
  ret = mpi->decode_put_packet(ctx, packet);
  if (ret != MPP_OK) {
    std::fprintf(stderr, "decode_put_packet %d\n", ret);
  }

  int empty_polls = 0;
  const int max_empty = 500;
  while (frame_count < max_frames + warmup) {
    MppFrame frame = NULL;
    auto t0 = clock::now();
    ret = mpi->decode_get_frame(ctx, &frame);
    auto t1 = clock::now();
    if (ret != MPP_OK || !frame) {
      empty_polls++;
      if (empty_polls >= max_empty) {
        std::fprintf(stderr, "decode_get_frame empty too long, stop\n");
        break;
      }
      // 微睡，避免忙等
      struct timespec ts{0, 1000 * 1000};  // 1ms
      nanosleep(&ts, nullptr);
      continue;
    }
    empty_polls = 0;

    if (mpp_frame_get_info_change(frame)) {
      RK_U32 w = mpp_frame_get_width(frame);
      RK_U32 h = mpp_frame_get_height(frame);
      RK_U32 hor = mpp_frame_get_hor_stride(frame);
      RK_U32 ver = mpp_frame_get_ver_stride(frame);
      std::printf("info change %ux%u stride %ux%u\n", w, h, hor, ver);
      if (frm_grp) {
        mpp_buffer_group_put(frm_grp);
        frm_grp = NULL;
      }
      mpp_buffer_group_get_internal(&frm_grp, MPP_BUFFER_TYPE_DRM);
      mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp);
      mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
      mpp_frame_deinit(&frame);
      continue;
    }

    if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
      mpp_frame_deinit(&frame);
      continue;
    }

    MppBuffer mbuf = mpp_frame_get_buffer(frame);
    if (!mbuf) {
      mpp_frame_deinit(&frame);
      continue;
    }

    int src_fd = mpp_buffer_get_fd(mbuf);
    int src_w = (int)mpp_frame_get_width(frame);
    int src_h = (int)mpp_frame_get_height(frame);
    int hor = (int)mpp_frame_get_hor_stride(frame);
    int ver = (int)mpp_frame_get_ver_stride(frame);

    float scale = std::min(model_w / (float)src_w, model_h / (float)src_h);
    int nw = (int)std::round(src_w * scale);
    int nh = (int)std::round(src_h * scale);
    int left = (model_w - nw) / 2;
    int top = (model_h - nh) / 2;

    auto t2 = clock::now();
    std::memset(input_mem->virt_addr, 114, (size_t)model_w * model_h * 3);
    rga_buffer_t src =
        wrapbuffer_fd(src_fd, src_w, src_h, RK_FORMAT_YCbCr_420_SP, hor, ver);
    rga_buffer_t dst = wrapbuffer_fd(input_mem->fd, model_w, model_h,
                                     RK_FORMAT_RGB_888, model_w, model_h);
    im_rect srect{0, 0, src_w, src_h};
    im_rect drect{left, top, nw, nh};
    im_rect prect{};
    rga_buffer_t pat{};
    IM_STATUS st =
        improcess(src, dst, pat, srect, drect, prect, IM_SYNC);
    auto t3 = clock::now();
    if (st != IM_STATUS_SUCCESS) {
      std::fprintf(stderr, "RGA failed: %s\n", imStrError(st));
      mpp_frame_deinit(&frame);
      return 1;
    }

    ret = rknn_run(rknn_ctx, nullptr);
    auto t4 = clock::now();
    if (ret != 0) {
      std::fprintf(stderr, "rknn_run failed %d\n", ret);
      mpp_frame_deinit(&frame);
      return 1;
    }

    if (frame_count >= warmup) {
      decode_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
      rga_ms.push_back(
          std::chrono::duration<double, std::milli>(t3 - t2).count());
      infer_ms.push_back(
          std::chrono::duration<double, std::milli>(t4 - t3).count());
      e2e_ms.push_back(
          std::chrono::duration<double, std::milli>(t4 - t0).count());
      measured++;
    }

    // cls max from last
    float* out0 = (float*)out_mems[0]->virt_addr;
    if (out_attrs[0].n_elems >= 84u * 8400u) {
      cls_max = 0.f;
      for (int a = 0; a < 8400; ++a)
        for (int c = 4; c < 84; ++c)
          cls_max = std::max(cls_max, out0[c * 8400 + a]);
    }

    int eos = mpp_frame_get_eos(frame);
    frame_count++;
    mpp_frame_deinit(&frame);
    if (eos) break;
  }

  auto dump = [](const char* name, std::vector<double> v) {
    double mean = v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::printf("  \"%s\": {\"p50\": %.3f, \"p99\": %.3f, \"mean\": %.3f},\n",
                name, percentile(v, 50), percentile(v, 99), mean);
  };

  std::printf("{\n");
  std::printf("  \"pipeline\": \"MPP_H264 -> RGA_NV12_letterbox -> RKNN_fd\",\n");
  std::printf("  \"note\": \"硬解零拷贝通路；rknn_set_io_mem 初始化绑一次\",\n");
  std::printf("  \"model_input\": \"%dx%d\",\n", model_w, model_h);
  std::printf("  \"warmup\": %d,\n", warmup);
  std::printf("  \"measured_frames\": %d,\n", measured);
  std::printf("  \"input_mem_fd\": %d,\n", input_mem->fd);
  std::printf("  \"cls_max_last\": %.4f,\n", cls_max);
  dump("decode_ms", decode_ms);
  dump("rga_preprocess_ms", rga_ms);
  dump("infer_ms", infer_ms);
  // last field no comma
  {
    auto v = e2e_ms;
    double mean =
        v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::printf("  \"e2e_ms\": {\"p50\": %.3f, \"p99\": %.3f, \"mean\": %.3f}\n",
                percentile(v, 50), percentile(v, 99), mean);
  }
  std::printf("}\n");

  for (auto* m : out_mems) rknn_destroy_mem(rknn_ctx, m);
  rknn_destroy_mem(rknn_ctx, input_mem);
  rknn_destroy(rknn_ctx);
  if (packet) mpp_packet_deinit(&packet);
  if (frm_grp) mpp_buffer_group_put(frm_grp);
  mpp_destroy(ctx);
  return measured > 0 ? 0 : 2;
}
