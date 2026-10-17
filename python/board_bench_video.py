#!/usr/bin/env python3
"""单路视频基线：本地文件 → OpenCV 解码 → CPU letterbox → RKNN → NMS。

口径（写进日志，面试时按这个讲）：
- e2e_ms      : 取到解码帧 → 结构化结果写出（含 decode 后处理、预处理、推理、NMS）
- infer_ms    : 仅 rknn.inference
- preprocess_ms / post_ms / decode_ms : 分段服务时间，不含排队（本脚本单线程无排队）
- 预热帧不计入分位统计

这是「CPU 预处理 + OpenCV 软解」基线，用来填阶段 1 数字；
MPP / RGA / dma-buf 对照另开脚本，不要和本基线混口径。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(out, scale, pad_x, pad_y, orig_w, orig_h, conf_thres, iou_thres):
    pred = np.squeeze(out).T
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    mask = scores >= conf_thres
    boxes_xywh, scores, cls_ids = boxes_xywh[mask], scores[mask], cls_ids[mask]
    if boxes_xywh.size == 0:
        return []

    cx, cy, w, h = boxes_xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)

    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        for k in nms(boxes[idx], scores[idx], iou_thres):
            i = idx[k]
            dets.append(
                {
                    "cls": int(c),
                    "score": float(scores[i]),
                    "xyxy": [float(x) for x in boxes[i].tolist()],
                }
            )
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    arr = np.sort(np.asarray(xs, dtype=np.float64))
    k = (len(arr) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(arr[f])
    return float(arr[f] * (c - k) + arr[c] * (k - f))


def run_mpstat(seconds: int) -> str | None:
    """采样 CPU；失败返回 None（板上无 sysstat 时跳过）。"""
    try:
        # 平均占用：100 - %idle
        out = subprocess.check_output(
            ["mpstat", "1", str(seconds)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def parse_mpstat_avg(text: str | None) -> float | None:
    if not text:
        return None
    # 取 Average 行最后一列 %idle（mpstat 默认）
    for line in text.splitlines():
        if "Average" in line or "平均时间" in line:
            parts = line.split()
            if len(parts) >= 11:
                try:
                    idle = float(parts[-1])
                    return round(100.0 - idle, 2)
                except ValueError:
                    continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/yolov8n_int8.rknn")
    ap.add_argument("--video", default="assets/bus_1080p30.mp4")
    ap.add_argument("--out-jsonl", default="bench/logs/single_stream.jsonl")
    ap.add_argument("--out-summary", default="bench/logs/single_stream_summary.json")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=0, help="0=整段视频")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--core-mask", type=int, default=0, help="0=默认；RKNNLite 核掩码若支持再填")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {args.video}")

    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        raise SystemExit("load_rknn failed")
    # NPU_CORE_0 = 1 等；0 表示不设置
    if args.core_mask:
        ret = rknn.init_runtime(core_mask=args.core_mask)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        raise SystemExit(f"init_runtime failed: {ret}")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    frame_id = 0
    measured = 0
    t_wall0 = time.perf_counter()

    # 同步采一段 CPU（时长粗估：按 max_frames 或默认 15s）
    cpu_proc = None
    cpu_seconds = 15
    try:
        cpu_proc = subprocess.Popen(
            ["mpstat", "1", str(cpu_seconds)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        cpu_proc = None

    with open(args.out_jsonl, "w", encoding="utf-8") as logf:
        while True:
            if args.max_frames and frame_id >= args.max_frames:
                break
            t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break

            t2 = time.perf_counter()
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lb, scale, pad_x, pad_y = letterbox(rgb, 640)
            inp = np.expand_dims(lb, 0)
            t3 = time.perf_counter()

            outs = rknn.inference(inputs=[inp])
            t4 = time.perf_counter()

            dets = postprocess(
                outs[0], scale, pad_x, pad_y, w, h, args.conf, args.iou
            )
            t5 = time.perf_counter()

            # 结构化结果「写出」：这里写到内存 dict，模拟产出
            result = {
                "stream_id": 0,
                "frame_id": frame_id,
                "ts_ms": (t5 - t_wall0) * 1000.0,
                "dets": dets,
            }
            _ = json.dumps(result)  # 计入写出成本的极小部分
            t6 = time.perf_counter()

            rec = {
                "frame_id": frame_id,
                "warmup": frame_id < args.warmup,
                "decode_ms": (t1 - t0) * 1000.0,
                "preprocess_ms": (t3 - t2) * 1000.0,
                "infer_ms": (t4 - t3) * 1000.0,
                "post_ms": (t5 - t4) * 1000.0,
                "e2e_ms": (t6 - t0) * 1000.0,
                "num_dets": len(dets),
                "width": w,
                "height": h,
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not rec["warmup"]:
                records.append(rec)
                measured += 1
            frame_id += 1

    cap.release()
    rknn.release()

    mpstat_text = None
    if cpu_proc is not None:
        try:
            mpstat_text, _ = cpu_proc.communicate(timeout=cpu_seconds + 5)
        except subprocess.TimeoutExpired:
            cpu_proc.kill()
            mpstat_text, _ = cpu_proc.communicate()

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in records]

    summary = {
        "pipeline": "opencv_decode + cpu_letterbox + rknn + cpu_nms",
        "note": "阶段1基线；非 MPP/RGA/dma-buf",
        "model": args.model,
        "video": args.video,
        "warmup_frames": args.warmup,
        "measured_frames": measured,
        "resolution": f"{records[0]['width']}x{records[0]['height']}" if records else None,
        "infer_ms": {
            "p50": percentile(col("infer_ms"), 50),
            "p99": percentile(col("infer_ms"), 99),
            "mean": float(np.mean(col("infer_ms"))) if records else None,
        },
        "e2e_ms": {
            "p50": percentile(col("e2e_ms"), 50),
            "p99": percentile(col("e2e_ms"), 99),
            "mean": float(np.mean(col("e2e_ms"))) if records else None,
        },
        "preprocess_ms": {
            "p50": percentile(col("preprocess_ms"), 50),
            "p99": percentile(col("preprocess_ms"), 99),
            "mean": float(np.mean(col("preprocess_ms"))) if records else None,
        },
        "decode_ms": {
            "p50": percentile(col("decode_ms"), 50),
            "mean": float(np.mean(col("decode_ms"))) if records else None,
        },
        "post_ms": {
            "p50": percentile(col("post_ms"), 50),
            "mean": float(np.mean(col("post_ms"))) if records else None,
        },
        "cpu_util_avg_pct": parse_mpstat_avg(mpstat_text),
        "mpstat_raw_tail": (mpstat_text or "").strip().splitlines()[-5:],
    }

    Path(args.out_summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\njsonl: {args.out_jsonl}")
    print(f"summary: {args.out_summary}")


if __name__ == "__main__":
    main()
