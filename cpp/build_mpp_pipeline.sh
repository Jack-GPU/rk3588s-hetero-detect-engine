#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/cpp/mpp_rga_rknn_bench"
g++ -O2 -std=c++17 \
  -I"${ROOT}/third_party/rknn/include" \
  -I"${ROOT}/third_party/librga/include" \
  -I"${ROOT}/third_party/mpp/include" \
  "${ROOT}/cpp/src/mpp_rga_rknn_bench.cpp" \
  -o "${OUT}" \
  -L"${ROOT}/third_party/librga/lib" \
  -L"${ROOT}/third_party/mpp/lib" \
  -Wl,-rpath,"${ROOT}/third_party/librga/lib" \
  -Wl,-rpath,"${ROOT}/third_party/mpp/lib" \
  -lrga -lrockchip_mpp -lrknnrt -ldl -lpthread
echo "built ${OUT}"
