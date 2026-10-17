#!/usr/bin/env python3
"""YOLOv8 ONNX → 混合 INT8 RKNN。

全 INT8 会把 [0,640] 的框和 [0,1] 的类别挤进同一量化尺度，类别被压成 0。
做法：hybrid step1 → 把输出相关层改成 float16 → step2 → export。
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN


FLOAT16_LAYERS = [
    "output0-rs",
    "output0",
    "output0_int8",
    "/model.22/Sigmoid_output_0_rs",
    "/model.22/Concat_1_output_0-rs",
    "/model.22/Concat_1_output_0_sw",
    "/model.22/Mul_2_output_0-rs",
    "/model.22/Concat_2_output_0-rs",
]


def _patch_quant_cfg(cfg_path: Path) -> list[str]:
    """在 custom_quantize_layers 里把检测输出相关层设为 float16。

    全 INT8 时 output0-rs 的量化尺度按框坐标(~640)标定，类别分数(~1)会被压成 0。
    """
    text = cfg_path.read_text(encoding="utf-8")
    block = "custom_quantize_layers:\n" + "".join(
        f"    {name}: float16\n" for name in FLOAT16_LAYERS
    )
    if re.match(r"(?s)^custom_quantize_layers:.*?(?=quantize_parameters:)", text):
        text = re.sub(
            r"(?s)^custom_quantize_layers:.*?(?=quantize_parameters:)",
            block,
            text,
            count=1,
        )
    elif "custom_quantize_layers: {}" in text:
        text = text.replace("custom_quantize_layers: {}", block.rstrip(), 1)
    else:
        text = block + "\n" + text
    cfg_path.write_text(text, encoding="utf-8")
    return list(FLOAT16_LAYERS)


def simulate(rknn: RKNN, image: Path) -> float:
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"读图失败: {image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, (640, 640)), 0)
    outs = rknn.inference(inputs=[img])
    pred = np.squeeze(outs[0])
    cls_max = float(pred[4:].max()) if pred.ndim == 2 else float(pred.max())
    print(f"simulate shape={pred.shape} cls_max={cls_max:.4f} box_max={float(pred[:4].max()):.2f}")
    return cls_max


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, default=Path("models/yolov8n.onnx"))
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/calib/coco_val300/dataset.txt"),
    )
    ap.add_argument("--out", type=Path, default=Path("models/yolov8n_int8.rknn"))
    ap.add_argument("--work", type=Path, default=Path("models/hybrid_work"))
    ap.add_argument("--simulate", type=Path, default=Path("assets/bus.jpg"))
    ap.add_argument("--target", default="rk3588")
    args = ap.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"找不到 ONNX: {args.onnx}")
    if not args.dataset.is_file():
        raise SystemExit(f"找不到 dataset: {args.dataset}")

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # step1 会在 CWD 生成 <stem>.model/.data/.quantization.cfg
    # 用 work 目录，避免污染仓库根
    work_onnx = args.work / args.onnx.name
    if work_onnx.resolve() != args.onnx.resolve():
        shutil.copy2(args.onnx, work_onnx)

    stem = work_onnx.stem  # yolov8n
    model_file = args.work / f"{stem}.model"
    data_file = args.work / f"{stem}.data"
    cfg_file = args.work / f"{stem}.quantization.cfg"

    # dataset 路径对 step1 用绝对路径更稳
    dataset_abs = str(args.dataset.resolve())

    print("========== hybrid step1 ==========")
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
        quantized_algorithm="normal",
        quantized_method="channel",
    )
    ret = rknn.load_onnx(model=str(work_onnx.resolve()))
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")

    # 在 work 目录下跑，让产物落在 work
    cwd_before = Path.cwd()
    try:
        import os

        os.chdir(args.work)
        ret = rknn.hybrid_quantization_step1(
            dataset=dataset_abs,
            proposal=False,
        )
    finally:
        os.chdir(cwd_before)
        rknn.release()

    if ret != 0:
        raise SystemExit(f"hybrid_quantization_step1 failed: {ret}")

    for p in (model_file, data_file, cfg_file):
        if not p.is_file():
            # 有时文件名跟 onnx 不完全一致，列一下
            print("work dir:", sorted(args.work.iterdir()))
            raise SystemExit(f"缺少 step1 产物: {p}")

    print("========== patch quantization.cfg ==========")
    changed = _patch_quant_cfg(cfg_file)
    print("float16 layers:", changed)
    print("--- cfg head ---")
    print("\n".join(cfg_file.read_text(encoding="utf-8").splitlines()[:80]))

    print("========== hybrid step2 + export ==========")
    rknn = RKNN(verbose=True)
    ret = rknn.hybrid_quantization_step2(
        model_input=str(model_file.resolve()),
        data_input=str(data_file.resolve()),
        model_quantization_cfg=str(cfg_file.resolve()),
    )
    if ret != 0:
        raise SystemExit(f"hybrid_quantization_step2 failed: {ret}")

    ret = rknn.export_rknn(str(args.out.resolve()))
    if ret != 0:
        raise SystemExit(f"export_rknn failed: {ret}")
    print(f"RKNN: {args.out} size={args.out.stat().st_size / 1024 / 1024:.1f} MB")

    if args.simulate.is_file():
        print("========== simulate ==========")
        ret = rknn.init_runtime()
        if ret != 0:
            raise SystemExit(f"init_runtime failed: {ret}")
        cls_max = simulate(rknn, args.simulate)
        if cls_max < 0.05:
            rknn.release()
            raise SystemExit(
                f"类别分数仍异常 (cls_max={cls_max:.4f})，请检查 quantization.cfg 里输出层是否真的是 float16"
            )
        print(f"OK: cls_max={cls_max:.4f}")

    rknn.release()
    print("DONE")


if __name__ == "__main__":
    main()
