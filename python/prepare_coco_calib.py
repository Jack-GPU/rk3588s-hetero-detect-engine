#!/usr/bin/env python3
"""从 COCO val2017 随机抽 N 张图，写成 RKNN 用的 dataset.txt。"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 官方标注 zip 在国内很慢；用 HF 上的 instances_val2017.json（约 20MB）。
ANN_URLS = [
    "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json",
    "https://hf-mirror.com/datasets/merve/coco/resolve/main/annotations/instances_val2017.json",
]
IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"


def download(url: str, path: Path, timeout: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "rk3588s-calib/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)


def download_ann(cache: Path) -> Path:
    dest = cache / "instances_val2017.json"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"use cached {dest}")
        return dest
    last_err: Exception | None = None
    for url in ANN_URLS:
        try:
            print(f"download {url}")
            download(url, dest, timeout=300)
            if dest.stat().st_size > 1_000_000:
                return dest
        except Exception as exc:  # noqa: BLE001 — 镜像失败则换下一个
            last_err = exc
            print(f"failed: {exc}")
            if dest.is_file():
                dest.unlink()
    raise SystemExit(f"无法下载 instances_val2017.json: {last_err}")


def fetch_one(name: str, dest: Path) -> Path:
    if not dest.is_file() or dest.stat().st_size == 0:
        download(IMG_URL.format(file_name=name), dest, timeout=60)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/calib/coco_val300"))
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache", type=Path, default=Path("data/calib/.cache"))
    args = parser.parse_args()

    ann_path = download_ann(args.cache)
    with ann_path.open(encoding="utf-8") as f:
        names = [img["file_name"] for img in json.load(f)["images"]]
    if args.num > len(names):
        raise SystemExit(f"val2017 只有 {len(names)} 张，无法抽 {args.num}")

    random.seed(args.seed)
    chosen = sorted(random.sample(names, args.num))
    args.out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, name, args.out / name): name for name in chosen
        }
        for fut in as_completed(futures):
            paths.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(chosen):
                print(f"progress {done}/{len(chosen)}")

    # RKNN 会把路径相对 dataset.txt 所在目录解析，这里只写文件名。
    lines_sorted = [p.name for p in sorted(paths)]
    dataset = args.out / "dataset.txt"
    dataset.write_text("\n".join(lines_sorted) + "\n", encoding="utf-8")
    print(f"wrote {dataset} ({len(lines_sorted)} images)")


if __name__ == "__main__":
    main()
