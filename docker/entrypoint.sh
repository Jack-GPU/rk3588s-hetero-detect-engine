#!/bin/bash
# 本项目容器入口。镜像里原来的入口会打印 rknn-dev 旧名，这里覆盖掉。
set -euo pipefail

if [ "$#" -eq 0 ] || [ "$1" = "bash" ]; then
  exec bash
fi

exec "$@"
