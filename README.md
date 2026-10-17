# RK3588S 端侧多路视频实时结构化系统

在 RK3588S 上跑通多路 1080p 实时检测：MPP 硬解 → RGA 缩放/色域转换 → RKNN 推理，全链路用 DMA-BUF 传 fd，CPU 不碰像素。

计划中的验收线（有实测再写进简历）：端到端 P99、CPU、整板功耗、热稳定。阶段划分见 [docs/实施计划.md](docs/实施计划.md)。算法上手（预训练 → ONNX → 混合 INT8 RKNN → 板上出框）见 [docs/算法上手.md](docs/算法上手.md)。

## 目录

```
├── docker/          PC 端开发容器（训练、导出 ONNX、转 RKNN）
├── python/          导出、校准、混合量化、板上推理脚本
├── cpp/             板端流水线骨架（MPP / RGA / RKNN / DRM）
├── configs/         数据集与类别配置
├── assets/          小样例图（可入库）
├── data/            校准图与数据集（本地生成，不入库）
├── models/          权重与 .rknn（本地生成，不入库）
├── scripts/         进容器
└── docs/            实施计划与算法上手
```

板端 Runtime（`librknnrt`）与 PC 上 RKNN Toolkit2 版本必须一致，本仓库锁定 **2.3.2**。

## 快速开始（PC）

需要 Docker。没有镜像时，`./scripts/dev.sh` 会构建 `rk3588s-video-struct:2.3.2`（含 Toolkit2 2.3.2、PyTorch、Ultralytics）。有 NVIDIA Container Runtime 时自动加 GPU。

```bash
./scripts/dev.sh                              # 进入容器，仓库挂到 /workspace
./scripts/dev.sh python python/check_env.py   # 环境自检
```

算法链路（容器内）：

```bash
# 1) 预训练权重 + 导出 ONNX
yolo predict model=yolov8n.pt source=assets/bus.jpg imgsz=640
python python/export_onnx.py --weights models/yolov8n.pt --out models/yolov8n.onnx

# 2) 校准集（学习用 30 张；正式基线用 COCO 300 张）
python python/prepare_calib.py --out data/calib --num 30
# 或：python python/prepare_coco_calib.py --out data/calib/coco_val300 --num 300

# 3) 混合 INT8（检测输出层保持 FP16，避免类别分数被量化成 0）
python python/convert_rknn.py \
  --onnx models/yolov8n.onnx \
  --dataset data/calib/dataset.txt \
  --out models/yolov8n_int8.rknn \
  --simulate assets/bus.jpg
```

## 板上验证

1. 板端 `librknnrt` 升到 **2.3.2**（与 Toolkit2 对齐）。
2. 安装 `rknn_toolkit_lite2==2.3.2`、`numpy`、`opencv-python-headless`（板子无外网时在 PC 下 wheel 再拷过去）。
3. 拷贝模型与脚本后推理：

```bash
scp models/yolov8n_int8.rknn python/board_infer_yolov8.py assets/bus.jpg root@板子IP:~/
ssh root@板子IP
python3 board_infer_yolov8.py \
  --model ~/yolov8n_int8.rknn \
  --image ~/bus.jpg \
  --out ~/bus_det.jpg
```

## 单路视频基线（阶段 1，已测）

路径：`OpenCV 软解 → CPU letterbox → RKNN 混合 INT8 → CPU NMS`（**尚非** MPP/RGA/dma-buf）。

| 指标 | 实测 |
| --- | --- |
| 分辨率 | 1 × 1080p |
| 样本 | 270 帧（预热 30） |
| 推理 P50 / P99 | 35.6 / 51.6 ms |
| 端到端 P50 / P99 | 61.1 / 124.7 ms |
| 预处理 P50 | 9.9 ms |
| CPU 平均 | ~14.6%（mpstat，采样窗内） |

原始摘要：[`bench/logs/single_stream_summary.json`](bench/logs/single_stream_summary.json)。复现见 [`docs/单路基线.md`](docs/单路基线.md)。

## 说明

- **不要用全 INT8 直接量化** Ultralytics 导出的 YOLOv8 单输出 ONNX：框坐标与类别分数量纲差两个数量级，类别会被压成 0。请走 `python/convert_rknn.py` 的混合量化。
- 权重、ONNX、RKNN、校准图、测试 mp4 体积大，已在 `.gitignore` 中排除；克隆后按文档本地生成即可。
- 下一步：RGA + `rknn_create_mem_from_fd` 做预处理对照（A/B），再谈四路与三核调度。
