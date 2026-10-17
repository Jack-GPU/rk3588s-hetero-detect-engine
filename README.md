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
## 预处理对照 A/B（已测）

同进程、1080p→640 letterbox，服务时间：

| 路径 | preprocess P50 | preprocess P99 |
| --- | ---: | ---: |
| A CPU letterbox + memcpy | 25.5 ms | 26.0 ms |
| B RGA 写入 `rknn_create_mem` 的 dma-buf fd | **5.3 ms** | 9.4 ms |

P50 约 **4.8×**。详见 [`docs/预处理对照.md`](docs/预处理对照.md) 与 [`bench/logs/preprocess_ab_summary.json`](bench/logs/preprocess_ab_summary.json)。

## 三核调度对照（已测）

4 路并发推理、同输入，仅计 `inference`：

| 策略 | 推理 P99 | 合计吞吐 T |
| --- | ---: | ---: |
| 单核争用 `CORE_0` | 114.7 ms | 39.4 fps |
| 静态绑核 0/1/2/0 | **64.0 ms**（C） | 74.5 fps |
| `NPU_CORE_AUTO` | **51.4 ms**（D） | **87.7 fps** |

本组负载下 AUTO 优于静态；按实测写。详见 [`docs/调度对照.md`](docs/调度对照.md)。

## 硬解零拷贝 MPP→RGA→RKNN（阶段 4 单路，已测）

路径：`MPP H.264 → RGA NV12 letterbox → RKNN dma-buf fd`（不含 CPU NMS）。

| 指标 | 软解基线 e2e | 硬解 e2e |
| --- | ---: | ---: |
| P50 | 61.1 ms | **27.8 ms** |
| P99 | 124.7 ms | **28.5 ms** |

硬解分段 P50：decode 0.34 / RGA 1.86 / infer 25.6 ms。软解 e2e 含 NMS；对齐后 decode+pre+infer 约 2.0×。详见 [`docs/硬解零拷贝.md`](docs/硬解零拷贝.md) 与 [`bench/logs/mpp_rga_rknn_summary.json`](bench/logs/mpp_rga_rknn_summary.json)。

下一步：四路 MPP 并发 + 有界队列/丢帧，补功耗与热稳定长测。
