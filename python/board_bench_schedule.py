#!/usr/bin/env python3
"""三核调度对照（同负载）。

策略（4 个并发推理上下文，模拟 4 路）：
  1) serial_core0  : 全部绑 NPU_CORE_0（单核串行争用）
  2) static_mask   : 流 0/1/2/3 → CORE_0 / CORE_1 / CORE_2 / CORE_0
  3) auto          : 全部 NPU_CORE_AUTO

口径：
  - infer_ms : 仅 rknn.inference 服务时间（输入已 letterbox 好，不含解码/预处理/NMS）
  - T_fps    : 四路合计成功完成的推理次数 / 墙钟时间
  - 预热不计入分位

用法（板上）：
  python3 board_bench_schedule.py --model yolov8n_int8.rknn --image bus.jpg
"""

from __future__ import annotations

import argparse
import json
import threading
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
    return canvas


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


def read_npu_load() -> str | None:
    path = Path("/sys/kernel/debug/rknpu/load")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def make_contexts(model: str, masks: list[int]) -> list[RKNNLite]:
    ctxs: list[RKNNLite] = []
    for m in masks:
        r = RKNNLite()
        if r.load_rknn(model) != 0:
            raise SystemExit(f"load_rknn failed mask={m}")
        if r.init_runtime(core_mask=m) != 0:
            raise SystemExit(f"init_runtime failed mask={m}")
        ctxs.append(r)
    return ctxs


def run_strategy(
    name: str,
    model: str,
    masks: list[int],
    inp: np.ndarray,
    frames_per_stream: int,
    warmup: int,
) -> dict:
    n_streams = len(masks)
    ctxs = make_contexts(model, masks)
    # 每路独立输入缓冲，避免伪共享争议
    inputs = [inp.copy() for _ in range(n_streams)]

    barrier = threading.Barrier(n_streams)
    results: dict[int, list[float]] = {i: [] for i in range(n_streams)}
    errors: list[str] = []
    lock = threading.Lock()

    def worker(sid: int) -> None:
        rknn = ctxs[sid]
        x = inputs[sid]
        try:
            for _ in range(warmup):
                rknn.inference(inputs=[x])
            barrier.wait()
            local: list[float] = []
            for _ in range(frames_per_stream):
                t0 = time.perf_counter()
                rknn.inference(inputs=[x])
                t1 = time.perf_counter()
                local.append((t1 - t0) * 1000.0)
            with lock:
                results[sid] = local
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"stream{sid}: {exc}")

    load_before = read_npu_load()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_streams)]
    t_wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_wall1 = time.perf_counter()
    load_after = read_npu_load()

    for r in ctxs:
        r.release()

    if errors:
        raise SystemExit(f"{name} failed: {errors}")

    all_ms: list[float] = []
    for sid in range(n_streams):
        all_ms.extend(results[sid])

    wall_s = t_wall1 - t_wall0
    total_done = n_streams * frames_per_stream
    summary = {
        "strategy": name,
        "masks": masks,
        "streams": n_streams,
        "frames_per_stream": frames_per_stream,
        "warmup_per_stream": warmup,
        "measured_infer_samples": len(all_ms),
        "infer_ms": {
            "p50": percentile(all_ms, 50),
            "p99": percentile(all_ms, 99),
            "mean": float(np.mean(all_ms)),
        },
        "per_stream_infer_p99_ms": {
            str(sid): percentile(results[sid], 99) for sid in range(n_streams)
        },
        "wall_s": wall_s,
        "T_fps": total_done / wall_s if wall_s > 0 else float("nan"),
        "npu_load_before": load_before,
        "npu_load_after": load_after,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/yolov8n_int8.rknn")
    ap.add_argument("--image", default="assets/bus.jpg")
    ap.add_argument("--frames", type=int, default=100, help="每路计量帧数（不含预热）")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="bench/logs/schedule_summary.json")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        # 允许直接喂 1080p rgb
        raise SystemExit(f"读图失败: {args.image}")
    rgb = cv2.cvtColor(cv2.resize(img, (1920, 1080)), cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(letterbox(rgb, 640), 0)

    strategies = [
        ("serial_core0", [RKNNLite.NPU_CORE_0] * 4),
        (
            "static_mask",
            [
                RKNNLite.NPU_CORE_0,
                RKNNLite.NPU_CORE_1,
                RKNNLite.NPU_CORE_2,
                RKNNLite.NPU_CORE_0,
            ],
        ),
        ("auto", [RKNNLite.NPU_CORE_AUTO] * 4),
    ]

    rows = []
    for name, masks in strategies:
        print(f"==> {name} masks={masks}", flush=True)
        row = run_strategy(name, args.model, masks, inp, args.frames, args.warmup)
        print(
            f"    infer P50/P99={row['infer_ms']['p50']:.2f}/{row['infer_ms']['p99']:.2f} ms  "
            f"T={row['T_fps']:.1f} fps",
            flush=True,
        )
        rows.append(row)

    # 简历符号：C=静态 P99，D=AUTO P99，T=静态吞吐（或表中各自写清）
    by = {r["strategy"]: r for r in rows}
    out = {
        "title": "NPU 3-core schedule compare (4 concurrent contexts)",
        "note": "同输入 tensor；仅计 inference；队列/丢帧未启用",
        "model": args.model,
        "input": "1920x1080 letterbox->640",
        "frames_per_stream": args.frames,
        "strategies": rows,
        "resume_symbols": {
            "C_static_infer_p99_ms": by["static_mask"]["infer_ms"]["p99"],
            "D_auto_infer_p99_ms": by["auto"]["infer_ms"]["p99"],
            "T_static_fps": by["static_mask"]["T_fps"],
            "T_auto_fps": by["auto"]["T_fps"],
            "T_serial_fps": by["serial_core0"]["T_fps"],
            "serial_infer_p99_ms": by["serial_core0"]["infer_ms"]["p99"],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["resume_symbols"], ensure_ascii=False, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
