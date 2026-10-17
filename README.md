# RK3588S 端侧多路视频实时结构化系统

在 RK3588S 上跑通 4 路 1080p 实时检测：MPP 硬解 → RGA 缩放/色域转换 → RKNN 推理 → DRM 显示，全链路用 DMA-BUF 传 fd，CPU 不碰像素。

验收线：端到端 P99 ≤ 120 ms，8 核平均 CPU ≤ 45%，整板功耗 ≤ 10 W，连续 2 小时不降频。

阶段和训练边界见 [docs/实施计划.md](docs/实施计划.md)。

## 目录

```
├── docker/          PC 端开发容器（训练、导出 ONNX、转 RKNN）
├── python/          训练、导出、量化脚本
├── cpp/             板端流水线（MPP / RGA / RKNN / DRM）
├── configs/         数据集与类别配置
├── data/            数据集与校准图（不入库）
├── models/          权重与 .rknn（不入库）
├── scripts/         进容器、环境自检
└── docs/            实施计划与后续测量记录
```

板端运行时（`rknnrt`、MPP、RGA）装在 RK3588S 上，不装进这个 x86 镜像。PC 镜像里的 RKNN Toolkit2 只负责转换和仿真，版本必须和板子上的 Runtime 一致，当前锁定 **2.3.2**。

## 开发环境

这台机器上已经有可用镜像，2026-10-17 复核通过，不需要重新构建：

本项目使用镜像 `rk3588s-video-struct:2.3.2`，容器名 `rk3588s-video-struct`。里面是 Toolkit2 2.3.2、PyTorch 2.1.2+cu121、Ultralytics、ONNX 1.15.0、`aarch64-linux-gnu-gcc` 11.4。

新机器只需先装好并启动 Docker Engine。没有该镜像时，`./scripts/dev.sh` 会用 `docker/Dockerfile` 构建 `rk3588s-video-struct:2.3.2`（首次约 10–20 分钟，需要能访问 `download.pytorch.org` 和 PyPI）。检测到 NVIDIA Container Runtime 时自动加上 GPU；没有则用 CPU，RKNN 转换仍然可用。

```bash
./scripts/dev.sh          # 进入容器，仓库挂到 /workspace
./scripts/dev.sh python python/check_env.py
```
