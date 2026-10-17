#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/cpp/mpp_multi_queue_bench"
g++ -O3 -std=c++17 \
  -I"${ROOT}/cpp/src" \
  -I"${ROOT}/third_party/rknn/include" \
  -I"${ROOT}/third_party/librga/include" \
  -I"${ROOT}/third_party/mpp/include" \
  "${ROOT}/cpp/src/mpp_multi_queue_bench.cpp" \
  -o "${OUT}" \
  -L"${ROOT}/third_party/librga/lib" \
  -L"${ROOT}/third_party/mpp/lib" \
  -Wl,-rpath,"${ROOT}/third_party/librga/lib" \
  -Wl,-rpath,"${ROOT}/third_party/mpp/lib" \
  -lrga -lrockchip_mpp -lrknnrt -ldl -lpthread
echo "built ${OUT}"
