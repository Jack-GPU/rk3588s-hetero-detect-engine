// 四路 MPP→RGA→RKNN + 有界队列（depth=1 丢旧帧）
//
// 口径：
//   - infer_ms / e2e_ms：成功推理帧的服务时间（入队时刻→infer 结束；e2e 含排队）
//   - drop_rate：被挤掉的旧帧 / (完成推理 + 丢弃)
//   - T_fps：四路合计成功推理次数 / 墙钟
//
// 用法：
//   ./mpp_multi_queue_bench model.rknn video.h264 [frames_per_stream=80] \
//       [auto|static|serial] [num_streams=4] [input_fps=30]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
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

using SteadyClock = std::chrono::steady_clock;

struct QueuedFrame {
  MppFrame frame = nullptr;
  SteadyClock::time_point t_enq{};
};

// depth=1：新帧到达时若槽位占用则丢弃旧帧
struct DropOldQueue {
  std::mutex mu;
  std::condition_variable cv;
  QueuedFrame slot{};
  bool has = false;
  bool closed = false;
  uint64_t drops = 0;
  uint64_t pushed = 0;

  void push_drop_old(MppFrame frame, SteadyClock::time_point t) {
    MppFrame to_drop = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (has && slot.frame) {
        to_drop = slot.frame;
        drops++;
      }
      slot.frame = frame;
      slot.t_enq = t;
      has = true;
      pushed++;
    }
    if (to_drop) mpp_frame_deinit(&to_drop);
    cv.notify_one();
  }

  // 返回 false 表示关闭且空
  bool pop(QueuedFrame& out) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return has || closed; });
    if (!has) return false;
    out = slot;
    slot = {};
    has = false;
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(mu);
      closed = true;
    }
    cv.notify_all();
  }
};

struct StreamCtx {
  int id = 0;
  rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
  rknn_context rknn = 0;
  rknn_tensor_mem* input_mem = nullptr;
  std::vector<rknn_tensor_mem*> out_mems;
  std::vector<rknn_tensor_attr> out_attrs;
  int model_w = 640;
  int model_h = 640;

  MppCtx mpp = nullptr;
  MppApi* mpi = nullptr;
  MppBufferGroup frm_grp = nullptr;
  MppPacket packet = nullptr;

  DropOldQueue q;
  std::vector<double> infer_ms;
  std::vector<double> e2e_ms;
  std::vector<double> rga_ms;
  std::mutex stats_mu;
  std::atomic<int> inferred{0};
  float cls_max = 0.f;
};

static std::mutex g_rga_mu;  // improcess 跨流串行，避免驱动竞态

static int init_rknn_stream(StreamCtx& s, const char* model_path) {
  int ret = rknn_init(&s.rknn, (void*)model_path, 0, 0, nullptr);
  if (ret != 0) return ret;
  ret = rknn_set_core_mask(s.rknn, s.mask);
  if (ret != 0) {
    std::fprintf(stderr, "stream%d set_core_mask %d failed %d\n", s.id,
                 (int)s.mask, ret);
    return ret;
  }

  rknn_tensor_attr in_attr{};
  in_attr.index = 0;
  rknn_query(s.rknn, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  in_attr.type = RKNN_TENSOR_UINT8;
  in_attr.fmt = RKNN_TENSOR_NHWC;
  s.model_w = (int)in_attr.dims[2];
  s.model_h = (int)in_attr.dims[1];
  s.input_mem = rknn_create_mem(s.rknn, in_attr.size_with_stride);
  rknn_set_io_mem(s.rknn, s.input_mem, &in_attr);

  rknn_input_output_num io_num{};
  rknn_query(s.rknn, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  s.out_mems.resize(io_num.n_output);
  s.out_attrs.resize(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    s.out_attrs[i].index = i;
    rknn_query(s.rknn, RKNN_QUERY_OUTPUT_ATTR, &s.out_attrs[i],
               sizeof(s.out_attrs[i]));
    s.out_attrs[i].type = RKNN_TENSOR_FLOAT32;
    s.out_mems[i] =
        rknn_create_mem(s.rknn, s.out_attrs[i].n_elems * sizeof(float));
    rknn_set_io_mem(s.rknn, s.out_mems[i], &s.out_attrs[i]);
  }
  return 0;
}

static int init_mpp_stream(StreamCtx& s, const uint8_t* data, size_t n) {
  int ret = mpp_create(&s.mpp, &s.mpi);
  if (ret != MPP_OK) return ret;
  ret = mpp_init(s.mpp, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
  if (ret != MPP_OK) return ret;
  RK_U32 need_split = 1;
  s.mpi->control(s.mpp, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
  mpp_packet_init(&s.packet, (void*)data, n);
  mpp_packet_set_eos(s.packet);
  ret = s.mpi->decode_put_packet(s.mpp, s.packet);
  return ret;
}

static void destroy_stream(StreamCtx& s) {
  s.q.close();
  {
    std::lock_guard<std::mutex> lk(s.q.mu);
    if (s.q.has && s.q.slot.frame) mpp_frame_deinit(&s.q.slot.frame);
    s.q.has = false;
  }
  for (auto* m : s.out_mems)
    if (m) rknn_destroy_mem(s.rknn, m);
  if (s.input_mem) rknn_destroy_mem(s.rknn, s.input_mem);
  if (s.rknn) rknn_destroy(s.rknn);
  if (s.packet) mpp_packet_deinit(&s.packet);
  if (s.frm_grp) mpp_buffer_group_put(s.frm_grp);
  if (s.mpp) mpp_destroy(s.mpp);
}

static void sleep_until(SteadyClock::time_point tp) {
  auto now = SteadyClock::now();
  if (tp <= now) return;
  auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tp - now).count();
  struct timespec ts{ns / 1000000000LL, ns % 1000000000LL};
  nanosleep(&ts, nullptr);
}

// 按 input_fps 限速送帧，模拟实时码流；depth=1 来不及推理则丢旧帧
static void decode_worker(StreamCtx* s, int target_push, double input_fps) {
  int pushed_ok = 0;
  int empty_polls = 0;
  const int max_empty = 800;
  const double period_s = input_fps > 0 ? (1.0 / input_fps) : 0.0;
  auto next_deadline = SteadyClock::now();
  while (pushed_ok < target_push) {
    MppFrame frame = nullptr;
    int ret = s->mpi->decode_get_frame(s->mpp, &frame);
    if (ret != MPP_OK || !frame) {
      empty_polls++;
      if (empty_polls >= max_empty) break;
      struct timespec ts{0, 500 * 1000};
      nanosleep(&ts, nullptr);
      continue;
    }
    empty_polls = 0;

    if (mpp_frame_get_info_change(frame)) {
      if (s->frm_grp) {
        mpp_buffer_group_put(s->frm_grp);
        s->frm_grp = nullptr;
      }
      mpp_buffer_group_get_internal(&s->frm_grp, MPP_BUFFER_TYPE_DRM);
      s->mpi->control(s->mpp, MPP_DEC_SET_EXT_BUF_GROUP, s->frm_grp);
      s->mpi->control(s->mpp, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
      mpp_frame_deinit(&frame);
      continue;
    }
    if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
      mpp_frame_deinit(&frame);
      continue;
    }
    if (!mpp_frame_get_buffer(frame)) {
      mpp_frame_deinit(&frame);
      continue;
    }

    int eos = mpp_frame_get_eos(frame);
    sleep_until(next_deadline);
    auto t_enq = SteadyClock::now();
    s->q.push_drop_old(frame, t_enq);
    pushed_ok++;
    if (period_s > 0) {
      next_deadline +=
          std::chrono::duration_cast<SteadyClock::duration>(
              std::chrono::duration<double>(period_s));
      // 落后太多则追当前时间，避免雪崩
      auto now = SteadyClock::now();
      if (next_deadline + std::chrono::milliseconds(100) < now)
        next_deadline = now;
    }
    if (eos) break;
  }
  s->q.close();
}

static bool rga_letterbox(StreamCtx* s, MppFrame frame) {
  MppBuffer mbuf = mpp_frame_get_buffer(frame);
  int src_fd = mpp_buffer_get_fd(mbuf);
  int src_w = (int)mpp_frame_get_width(frame);
  int src_h = (int)mpp_frame_get_height(frame);
  int hor = (int)mpp_frame_get_hor_stride(frame);
  int ver = (int)mpp_frame_get_ver_stride(frame);
  float scale =
      std::min(s->model_w / (float)src_w, s->model_h / (float)src_h);
  int nw = (int)std::round(src_w * scale);
  int nh = (int)std::round(src_h * scale);
  int left = (s->model_w - nw) / 2;
  int top = (s->model_h - nh) / 2;

  std::lock_guard<std::mutex> rga_lk(g_rga_mu);
  std::memset(s->input_mem->virt_addr, 114,
              (size_t)s->model_w * s->model_h * 3);
  rga_buffer_t src =
      wrapbuffer_fd(src_fd, src_w, src_h, RK_FORMAT_YCbCr_420_SP, hor, ver);
  rga_buffer_t dst =
      wrapbuffer_fd(s->input_mem->fd, s->model_w, s->model_h, RK_FORMAT_RGB_888,
                    s->model_w, s->model_h);
  im_rect srect{0, 0, src_w, src_h};
  im_rect drect{left, top, nw, nh};
  im_rect prect{};
  rga_buffer_t pat{};
  IM_STATUS st = improcess(src, dst, pat, srect, drect, prect, IM_SYNC);
  return st == IM_STATUS_SUCCESS;
}

static void infer_worker(StreamCtx* s, int warmup, int measure_target,
                         std::atomic<int>* warmup_done) {
  int seen = 0;
  bool signaled_warmup = false;
  while (true) {
    QueuedFrame qf;
    if (!s->q.pop(qf)) break;
    if (!qf.frame) continue;

    auto t_rga0 = SteadyClock::now();
    bool ok = rga_letterbox(s, qf.frame);
    auto t_rga1 = SteadyClock::now();
    if (!ok) {
      std::fprintf(stderr, "stream%d RGA failed\n", s->id);
      mpp_frame_deinit(&qf.frame);
      continue;
    }

    auto t_inf0 = SteadyClock::now();
    int ret = rknn_run(s->rknn, nullptr);
    auto t_inf1 = SteadyClock::now();
    mpp_frame_deinit(&qf.frame);

    if (ret != 0) {
      std::fprintf(stderr, "stream%d rknn_run %d\n", s->id, ret);
      continue;
    }

    seen++;
    if (seen == warmup && !signaled_warmup) {
      warmup_done->fetch_add(1);
      signaled_warmup = true;
    }

    if (seen > warmup && s->inferred.load() < measure_target) {
      double infer =
          std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();
      double rga =
          std::chrono::duration<double, std::milli>(t_rga1 - t_rga0).count();
      double e2e =
          std::chrono::duration<double, std::milli>(t_inf1 - qf.t_enq).count();
      {
        std::lock_guard<std::mutex> lk(s->stats_mu);
        s->infer_ms.push_back(infer);
        s->rga_ms.push_back(rga);
        s->e2e_ms.push_back(e2e);
      }
      s->inferred.fetch_add(1);
    }

    if (!s->out_mems.empty() && s->out_attrs[0].n_elems >= 84u * 8400u) {
      float* out0 = (float*)s->out_mems[0]->virt_addr;
      float mx = 0.f;
      for (int a = 0; a < 8400; ++a)
        for (int c = 4; c < 84; ++c) mx = std::max(mx, out0[c * 8400 + a]);
      s->cls_max = mx;
    }
  }
  if (!signaled_warmup) warmup_done->fetch_add(1);
}

static std::vector<rknn_core_mask> masks_for_mode(const std::string& mode,
                                                  int n) {
  std::vector<rknn_core_mask> m(n, RKNN_NPU_CORE_AUTO);
  if (mode == "serial") {
    for (int i = 0; i < n; ++i) m[i] = RKNN_NPU_CORE_0;
  } else if (mode == "static") {
    rknn_core_mask table[] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
                              RKNN_NPU_CORE_0};
    for (int i = 0; i < n; ++i) m[i] = table[i % 4];
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s model.rknn video.h264 [frames_per_stream] "
                 "[auto|static|serial] [num_streams] [input_fps]\n",
                 argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* h264_path = argv[2];
  int frames_per = argc > 3 ? std::atoi(argv[3]) : 80;
  std::string mode = argc > 4 ? argv[4] : "auto";
  int num_streams = argc > 5 ? std::atoi(argv[5]) : 4;
  double input_fps = argc > 6 ? std::atof(argv[6]) : 30.0;
  if (num_streams < 1 || num_streams > 8) num_streams = 4;
  const int warmup = 15;
  // 丢帧时仍按 fps 推进，多送一轮保证测满 frames_per
  int target_push = (frames_per + warmup) * 2;

  auto bitstream = read_file(h264_path);
  if (bitstream.empty()) {
    std::fprintf(stderr, "read h264 failed\n");
    return 1;
  }
  // 拼接多段 annex-B，避免限速长测时过早 EOS
  std::vector<uint8_t> long_bs;
  long_bs.reserve(bitstream.size() * 3);
  for (int i = 0; i < 3; ++i)
    long_bs.insert(long_bs.end(), bitstream.begin(), bitstream.end());

  auto masks = masks_for_mode(mode, num_streams);
  std::vector<StreamCtx> streams(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    streams[i].id = i;
    streams[i].mask = masks[i];
    if (init_rknn_stream(streams[i], model_path) != 0) {
      std::fprintf(stderr, "init_rknn stream%d failed\n", i);
      return 1;
    }
    if (init_mpp_stream(streams[i], long_bs.data(), long_bs.size()) !=
        MPP_OK) {
      std::fprintf(stderr, "init_mpp stream%d failed\n", i);
      return 1;
    }
  }

  std::atomic<int> warmup_done{0};

  std::vector<std::thread> threads;
  threads.reserve(num_streams * 2);
  for (int i = 0; i < num_streams; ++i) {
    threads.emplace_back(infer_worker, &streams[i], warmup, frames_per,
                         &warmup_done);
  }
  for (int i = 0; i < num_streams; ++i) {
    threads.emplace_back(decode_worker, &streams[i], target_push, input_fps);
  }

  for (int wait_i = 0; warmup_done.load() < num_streams && wait_i < 60000;
       ++wait_i) {
    struct timespec ts{0, 500 * 1000};
    nanosleep(&ts, nullptr);
  }
  if (warmup_done.load() < num_streams) {
    std::fprintf(stderr, "warmup timeout (%d/%d)\n", warmup_done.load(),
                 num_streams);
  }
  auto t_wall0 = SteadyClock::now();
  for (auto& t : threads) t.join();
  auto t_wall1 = SteadyClock::now();
  double wall_s =
      std::chrono::duration<double>(t_wall1 - t_wall0).count();

  uint64_t total_infer = 0, total_drop = 0, total_push = 0;
  std::vector<double> all_infer, all_e2e, all_rga;
  for (auto& s : streams) {
    total_infer += s.inferred.load();
    total_drop += s.q.drops;
    total_push += s.q.pushed;
    std::lock_guard<std::mutex> lk(s.stats_mu);
    all_infer.insert(all_infer.end(), s.infer_ms.begin(), s.infer_ms.end());
    all_e2e.insert(all_e2e.end(), s.e2e_ms.begin(), s.e2e_ms.end());
    all_rga.insert(all_rga.end(), s.rga_ms.begin(), s.rga_ms.end());
  }
  double drop_rate =
      (total_infer + total_drop) > 0
          ? (double)total_drop / (double)(total_infer + total_drop)
          : 0.0;
  // T：只计测量期成功推理（各流 inferred）
  double T_fps = wall_s > 0 ? (double)total_infer / wall_s : 0.0;

  auto dump = [](const char* name, const std::vector<double>& v) {
    double mean =
        v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double p50 = percentile(v, 50);
    double p99 = percentile(v, 99);
    if (v.empty()) {
      std::printf("  \"%s\": {\"p50\": null, \"p99\": null, \"mean\": 0.0}",
                  name);
    } else {
      std::printf("  \"%s\": {\"p50\": %.3f, \"p99\": %.3f, \"mean\": %.3f}",
                  name, p50, p99, mean);
    }
  };

  std::printf("{\n");
  std::printf("  \"pipeline\": \"4x MPP_H264 -> queue(depth=1 drop-old) -> "
              "RGA -> RKNN_fd\",\n");
  std::printf("  \"mode\": \"%s\",\n", mode.c_str());
  std::printf("  \"num_streams\": %d,\n", num_streams);
  std::printf("  \"input_fps\": %.1f,\n", input_fps);
  std::printf("  \"queue_depth\": 1,\n");
  std::printf("  \"drop_policy\": \"drop_old\",\n");
  std::printf("  \"core_masks\": [");
  for (int i = 0; i < num_streams; ++i)
    std::printf("%s%d", i ? ", " : "", (int)masks[i]);
  std::printf("],\n");
  std::printf("  \"warmup_per_stream\": %d,\n", warmup);
  std::printf("  \"frames_per_stream_target\": %d,\n", frames_per);
  std::printf("  \"wall_s\": %.3f,\n", wall_s);
  std::printf("  \"total_inferred\": %llu,\n",
              (unsigned long long)total_infer);
  std::printf("  \"total_pushed\": %llu,\n", (unsigned long long)total_push);
  std::printf("  \"total_drops\": %llu,\n", (unsigned long long)total_drop);
  std::printf("  \"drop_rate\": %.4f,\n", drop_rate);
  std::printf("  \"T_fps\": %.2f,\n", T_fps);
  std::printf("  \"cls_max_last_stream0\": %.4f,\n", streams[0].cls_max);
  dump("infer_ms_all", all_infer);
  std::printf(",\n");
  dump("rga_ms_all", all_rga);
  std::printf(",\n");
  dump("e2e_ms_all", all_e2e);
  std::printf(",\n");
  std::printf("  \"per_stream\": [\n");
  for (int i = 0; i < num_streams; ++i) {
    auto& s = streams[i];
    std::lock_guard<std::mutex> lk(s.stats_mu);
    auto fmt = [](double x) {
      char buf[32];
      if (std::isnan(x))
        std::snprintf(buf, sizeof(buf), "null");
      else
        std::snprintf(buf, sizeof(buf), "%.3f", x);
      return std::string(buf);
    };
    std::printf(
        "    {\"id\": %d, \"mask\": %d, \"inferred\": %d, \"drops\": %llu, "
        "\"infer_p50\": %s, \"infer_p99\": %s, \"e2e_p50\": %s, "
        "\"e2e_p99\": %s}%s\n",
        s.id, (int)s.mask, s.inferred.load(), (unsigned long long)s.q.drops,
        fmt(percentile(s.infer_ms, 50)).c_str(),
        fmt(percentile(s.infer_ms, 99)).c_str(),
        fmt(percentile(s.e2e_ms, 50)).c_str(),
        fmt(percentile(s.e2e_ms, 99)).c_str(),
        i + 1 < num_streams ? "," : "");
  }
  std::printf("  ]\n");
  std::printf("}\n");

  for (auto& s : streams) destroy_stream(s);
  return total_infer > 0 ? 0 : 2;
}
