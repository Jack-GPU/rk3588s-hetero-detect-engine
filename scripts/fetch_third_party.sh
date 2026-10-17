#!/usr/bin/env bash
# 下载板上编译 A/B 对照所需的 librga 与 rknn_api.h。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party/librga/lib" "$ROOT/third_party/librga/include" \
         "$ROOT/third_party/rknn/include"

wget -q -O "$ROOT/third_party/librga/lib/librga.so" \
  "https://raw.githubusercontent.com/airockchip/librga/main/libs/Linux/gcc-aarch64/librga.so"
wget -q -O "$ROOT/third_party/rknn/include/rknn_api.h" \
  "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h"
wget -q -O "$ROOT/third_party/stb_image.h" \
  "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"

for f in im2d.h im2d.hpp im2d_buffer.h im2d_common.h im2d_expand.h im2d_mpi.h \
         im2d_single.h im2d_task.h im2d_type.h im2d_version.h rga.h drmrga.h \
         RgaApi.h RgaUtils.h; do
  wget -q -O "$ROOT/third_party/librga/include/$f" \
    "https://raw.githubusercontent.com/airockchip/librga/main/include/$f"
done
echo "ok: third_party ready"
