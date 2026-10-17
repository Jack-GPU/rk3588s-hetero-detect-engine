# 从零构建本项目镜像，标签：rk3588s-video-struct:2.3.2
# 范围内的组件：RKNN Toolkit2 2.3.2、PyTorch CUDA 11.8/12.1 训练、Ultralytics、ONNX、aarch64 交叉编译。
# 板端 rknnrt / MPP / RGA 不在此镜像内。
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    YOLO_CONFIG_DIR=/tmp \
    PYTHONUNBUFFERED=1

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential cmake pkg-config ninja-build \
        gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libjpeg-dev libpng-dev zlib1g-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 版本与 RKNN Toolkit2 2.3.2 对齐。torch 用 CUDA 12.1 wheel，容器需 --gpus all。
RUN python -m pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install --no-cache-dir \
        numpy \
        onnx==1.15.0 \
        onnxruntime==1.16.3 \
        onnxsim \
        opencv-python-headless \
        ultralytics==8.4.177 \
        rknn-toolkit2==2.3.2

WORKDIR /workspace
CMD ["/bin/bash"]
