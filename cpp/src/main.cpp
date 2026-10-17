// 板端入口说明：完整流水线以专项 bench 交付（可复现数字）。
// 构建：cpp/build_mpp_pipeline.sh / cpp/build_mpp_multi_queue.sh
#include <cstdio>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::printf(
      "rk3588s-video-struct board entry\n"
      "\n"
      "Bench binaries (build on board with matching scripts):\n"
      "  cpp/mpp_rga_rknn_bench     single-stream MPP→RGA→RKNN→NMS\n"
      "  cpp/mpp_multi_queue_bench  4-stream + depth-1 drop-old queue\n"
      "  cpp/preprocess_ab_bench    CPU vs RGA preprocess A/B\n"
      "\n"
      "Python:\n"
      "  board_bench_video.py / board_bench_schedule.py\n"
      "  board_soak_power.py / eval_map_coco.py\n"
      "\n"
      "See README.md and docs/ for measured numbers.\n");
  return 0;
}
