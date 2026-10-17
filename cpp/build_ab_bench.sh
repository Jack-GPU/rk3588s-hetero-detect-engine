#!/usr/bin/env bash
# 在 RK3588S 板上编译预处理 A/B 对照程序。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/cpp/preprocess_ab_bench"
g++ -O2 -std=c++17 \
  -I"${ROOT}/third_party/rknn/include" \
  -I"${ROOT}/third_party/librga/include" \
  -I"${ROOT}/third_party" \
  "${ROOT}/cpp/src/preprocess_ab_bench.cpp" \
  -o "${OUT}" \
  -L"${ROOT}/third_party/librga/lib" -Wl,-rpath,"${ROOT}/third_party/librga/lib" \
  -lrga -lrknnrt -ldl -lpthread
echo "built ${OUT}"
