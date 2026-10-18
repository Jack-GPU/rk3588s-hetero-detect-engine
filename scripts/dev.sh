#!/usr/bin/env bash
# 启动并进入本项目容器 rk3588s-hetero-detect-engine。
# 无参数时打开 bash；有参数时在容器内执行。
# 例：./scripts/dev.sh python python/check_env.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/docker"
IMAGE="rk3588s-hetero-detect-engine:2.3.2"

if ! docker info >/dev/null 2>&1; then
  echo "Docker 守护进程不可用。请先安装并启动 Docker Engine：" >&2
  echo "  https://docs.docker.com/engine/install/" >&2
  exit 1
fi

compose() {
  local files=(-f docker-compose.yml)
  if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
    files+=(-f docker-compose.gpu.yml)
  else
    echo "未检测到 NVIDIA Container Runtime，容器以 CPU 启动。RKNN 转换可用，训练会慢。" >&2
    echo "需要 GPU 时安装驱动和 nvidia-container-toolkit 后重进容器。" >&2
  fi
  docker compose "${files[@]}" "$@"
}

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "未找到 ${IMAGE}，开始从 docker/Dockerfile 构建。"
  echo "首次构建要下载 PyTorch CUDA wheel 和 RKNN Toolkit2，大约 10–20 分钟。"
  docker build -t "${IMAGE}" -f Dockerfile .
fi

compose up -d dev

if [ "$#" -eq 0 ]; then
  compose exec dev bash
else
  compose exec dev "$@"
fi
