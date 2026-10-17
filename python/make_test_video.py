#!/usr/bin/env python3
"""从一张图生成固定码率风格的 1080p30 测试短片（板上/PC 均可）。

用途：单路基线延迟测量。不是真实场景数据，只保证分辨率和帧数稳定、可复现。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=Path("assets/bus.jpg"))
    ap.add_argument("--out", type=Path, default=Path("assets/bus_1080p30.mp4"))
    ap.add_argument("--frames", type=int, default=300, help="总帧数，默认 10s@30fps")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"读图失败: {args.image}")

    base = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # mp4v 兼容性最好；板上 OpenCV 通常自带
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打开失败，检查 OpenCV 是否带视频编码")

    h, w = base.shape[:2]
    for i in range(args.frames):
        # 轻微平移，避免编解码器把所有帧当成静止帧极端压缩
        dx = int(8 * np.sin(i / 15.0))
        dy = int(4 * np.cos(i / 21.0))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(
            base, m, (w, h), borderMode=cv2.BORDER_REFLECT_101
        )
        writer.write(frame)
    writer.release()

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(
        f"saved {args.out}  {args.width}x{args.height}@{args.fps}  "
        f"frames={args.frames}  size={size_mb:.1f}MB"
    )


if __name__ == "__main__":
    main()
