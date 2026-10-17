#!/usr/bin/env python3
"""把 YOLOv8n.pt 导出为固定形状的ONNX，供 RKNN Toolkit2 使用。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--out", type=Path, default=Path("models/yolov8n.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"找不到权重: {args.weights}（先按 docs/算法上手.md 第 1 步下载）")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        batch=1,
    )
    exported_path = Path(exported)
    if exported_path.resolve() != args.out.resolve():
        args.out.write_bytes(exported_path.read_bytes())
        print(f"已复制到 {args.out}")
    print(f"ONNX: {args.out}  size={args.out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
