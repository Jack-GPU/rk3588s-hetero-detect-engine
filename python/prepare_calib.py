#!/usr/bin/env python3
"""下载若干张公开图，写成 RKNN 量化用的 dataset.txt。"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# 学习用小集合；正式基线请换成自己的场景图或 COCO val。
URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/calib"))
    parser.add_argument("--num", type=int, default=30, help="目标张数（不足则循环复用 URL）")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for i in range(args.num):
        url = URLS[i % len(URLS)]
        path = args.out / f"{i:04d}.jpg"
        if not path.is_file():
            print(f"download {url} -> {path}")
            urllib.request.urlretrieve(url, path)
        lines.append(str(path.resolve()))

    dataset = args.out / "dataset.txt"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dataset} ({len(lines)} images)")


if __name__ == "__main__":
    main()
