#!/usr/bin/env python3
"""COCO val 子集上对比 FP32（Ultralytics）与板端 INT8（RKNNLite）的 mAP50。

在 PC / 开发容器跑 FP32；在板上跑 INT8 并写出预测 JSON，再回 PC 算 mAP。

示例：
  # PC：下 100 张 val2017，跑 FP32
  python python/eval_map_coco.py prepare --out data/coco_val100 --num 100
  python python/eval_map_coco.py fp32 --images data/coco_val100 --weights models/yolov8n.pt

  # 板：跑 INT8
  python3 eval_map_coco.py int8 --images coco_val100 --model yolov8n_int8.rknn --out pred_int8.json

  # PC：汇总
  python python/eval_map_coco.py summarize --fp32 runs/map_fp32.json --int8 pred_int8.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def letterbox(img, size=640, color=(114, 114, 114)):
    import cv2

    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
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


def postprocess(out, scale, pad_x, pad_y, orig_w, orig_h, conf=0.25, iou=0.45):
    pred = np.squeeze(out).T
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    mask = scores >= conf
    boxes_xywh, scores, cls_ids = boxes_xywh[mask], scores[mask], cls_ids[mask]
    if boxes_xywh.size == 0:
        return []
    cx, cy, w, h = boxes_xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)
    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        for k in nms(boxes[idx], scores[idx], iou):
            i = idx[k]
            x1, y1, x2, y2 = boxes[i].tolist()
            dets.append(
                {
                    "category_id": int(c),  # ultralytics 0-based；COCO 需映射
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[i]),
                }
            )
    return dets


def cmd_prepare(args):
    import random

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    val_zip = cache / "val2017.zip"
    ann_zip = cache / "annotations_trainval2017.zip"
    if not val_zip.exists():
        print(f"download {COCO_VAL_URL}")
        urllib.request.urlretrieve(COCO_VAL_URL, val_zip)
    if not ann_zip.exists():
        print(f"download {COCO_ANN_URL}")
        urllib.request.urlretrieve(COCO_ANN_URL, ann_zip)

    with zipfile.ZipFile(ann_zip) as zf:
        zf.extract("annotations/instances_val2017.json", cache)
    ann_path = cache / "annotations/instances_val2017.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    images = ann["images"]
    random.seed(args.seed)
    random.shuffle(images)
    pick = images[: args.num]
    ids = {im["id"] for im in pick}
    with zipfile.ZipFile(val_zip) as zf:
        for im in pick:
            name = im["file_name"]
            src = f"val2017/{name}"
            dst = img_dir / name
            if not dst.exists():
                dst.write_bytes(zf.read(src))
    sub = {
        "info": ann.get("info", {}),
        "licenses": ann.get("licenses", []),
        "images": pick,
        "annotations": [a for a in ann["annotations"] if a["image_id"] in ids],
        "categories": ann["categories"],
    }
    (out / "instances.json").write_text(json.dumps(sub), encoding="utf-8")
    # ultralytics / COCO80 类别名 → coco cat id
    (out / "manifest.json").write_text(
        json.dumps({"num": len(pick), "images": str(img_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"prepared {len(pick)} images -> {out}")


def _coco80_to_coco_cat():
    # Ultralytics 0..79 → COCO category_id
    from ultralytics.data.utils import YOLO_COCO80_TO_COCO91

    # package helper may vary; fallback table
    try:
        return list(YOLO_COCO80_TO_COCO91)
    except Exception:
        # standard mapping used by ultralytics
        return [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90,
        ]


def cmd_fp32(args):
    import cv2
    from ultralytics import YOLO

    images = sorted(Path(args.images).joinpath("images").glob("*.jpg"))
    if not images:
        images = sorted(Path(args.images).glob("*.jpg"))
    model = YOLO(args.weights)
    mapping = _coco80_to_coco_cat()
    preds = []
    # need image_id from instances.json
    inst = Path(args.images) / "instances.json"
    name_to_id = {}
    if inst.exists():
        for im in json.loads(inst.read_text())["images"]:
            name_to_id[im["file_name"]] = im["id"]

    for p in images:
        r = model.predict(str(p), imgsz=640, conf=args.conf, iou=args.iou, verbose=False)[0]
        image_id = name_to_id.get(p.name)
        if image_id is None:
            continue
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls.item())
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            preds.append(
                {
                    "image_id": image_id,
                    "category_id": mapping[cls],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(b.conf.item()),
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds), encoding="utf-8")
    print(f"fp32 preds {len(preds)} -> {out}")
    _print_map(Path(args.images) / "instances.json", out, tag="fp32")


def cmd_int8(args):
    import cv2
    from rknnlite.api import RKNNLite

    images = sorted(Path(args.images).joinpath("images").glob("*.jpg"))
    if not images:
        images = sorted(Path(args.images).glob("*.jpg"))
    inst = Path(args.images) / "instances.json"
    name_to_id = {}
    if inst.exists():
        for im in json.loads(inst.read_text())["images"]:
            name_to_id[im["file_name"]] = im["id"]
    mapping = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90,
    ]

    rknn = RKNNLite()
    assert rknn.load_rknn(args.model) == 0
    assert rknn.init_runtime() == 0
    preds = []
    for p in images:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lb, scale, pad_x, pad_y = letterbox(rgb, 640)
        outs = rknn.inference(inputs=[np.expand_dims(lb, 0)])
        dets = postprocess(outs[0], scale, pad_x, pad_y, w, h, args.conf, args.iou)
        image_id = name_to_id.get(p.name)
        if image_id is None:
            continue
        for d in dets:
            d = dict(d)
            d["image_id"] = image_id
            d["category_id"] = mapping[d["category_id"]]
            preds.append(d)
    rknn.release()
    out = Path(args.out)
    out.write_text(json.dumps(preds), encoding="utf-8")
    print(f"int8 preds {len(preds)} -> {out}")
    if inst.exists():
        _print_map(inst, out, tag="int8")


def _print_map(ann_path: Path, pred_path: Path, tag: str):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools 未安装，跳过 mAP 计算；预测已保存。")
        return
    if not ann_path.exists():
        print("无 instances.json，跳过 mAP")
        return
    coco = COCO(str(ann_path))
    preds = json.loads(Path(pred_path).read_text())
    if not preds:
        print(f"{tag}: empty preds")
        return
    dt = coco.loadRes(preds)
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # AP @0.50 is stats[1]
    summary = {
        "tag": tag,
        "mAP50_95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "num_preds": len(preds),
    }
    out = Path(pred_path).with_name(f"map_{tag}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_summarize(args):
    fp = json.loads(Path(args.fp32).read_text()) if Path(args.fp32).exists() else {}
    ip = json.loads(Path(args.int8).read_text()) if Path(args.int8).exists() else {}
    # allow either pred json or map_*.json
    def load_map(path: str):
        p = Path(path)
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        if isinstance(j, dict) and "mAP50" in j:
            return j
        # sibling map file
        for tag in ("fp32", "int8"):
            m = p.with_name(f"map_{tag}.json")
            if m.exists():
                return json.loads(m.read_text())
        return None

    fpm = load_map(args.fp32) or load_map(str(Path(args.fp32).with_name("map_fp32.json")))
    ipm = load_map(args.int8) or load_map(str(Path(args.int8).with_name("map_int8.json")))
    out = {
        "fp32": fpm,
        "int8": ipm,
        "mAP50_drop": None
        if not (fpm and ipm)
        else round(fpm["mAP50"] - ipm["mAP50"], 4),
        "note": "掉点 = FP32 mAP50 - INT8 mAP50；同一 COCO 子集",
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("prepare")
    p0.add_argument("--out", default="data/coco_val100")
    p0.add_argument("--cache", default="data/coco_cache")
    p0.add_argument("--num", type=int, default=100)
    p0.add_argument("--seed", type=int, default=0)
    p0.set_defaults(func=cmd_prepare)

    p1 = sub.add_parser("fp32")
    p1.add_argument("--images", default="data/coco_val100")
    p1.add_argument("--weights", default="models/yolov8n.pt")
    p1.add_argument("--out", default="runs/map_fp32_preds.json")
    p1.add_argument("--conf", type=float, default=0.25)
    p1.add_argument("--iou", type=float, default=0.45)
    p1.set_defaults(func=cmd_fp32)

    p2 = sub.add_parser("int8")
    p2.add_argument("--images", default="data/coco_val100")
    p2.add_argument("--model", default="yolov8n_int8.rknn")
    p2.add_argument("--out", default="pred_int8.json")
    p2.add_argument("--conf", type=float, default=0.25)
    p2.add_argument("--iou", type=float, default=0.45)
    p2.set_defaults(func=cmd_int8)

    p3 = sub.add_parser("summarize")
    p3.add_argument("--fp32", default="runs/map_fp32.json")
    p3.add_argument("--int8", default="map_int8.json")
    p3.add_argument("--out", default="bench/logs/map_compare.json")
    p3.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
