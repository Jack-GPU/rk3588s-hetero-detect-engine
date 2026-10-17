#!/usr/bin/env python3
"""板上用 RKNNLite 跑 YOLOv8n INT8：读图 → 推理 → 画框保存。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

COCO80 = (
    "person bicycle car motorcycle airplane bus train truck boat traffic light "
    "fire hydrant stop sign parking meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports ball kite baseball bat baseball glove skateboard "
    "surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch "
    "potted plant bed dining table toilet tv laptop mouse remote keyboard "
    "cell phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy bear hair drier toothbrush"
).split()


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


def postprocess(
    out: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
    conf_thres: float,
    iou_thres: float,
):
    # out: (1, 84, 8400) → (8400, 84)
    pred = np.squeeze(out).T
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    mask = scores >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    if boxes_xywh.size == 0:
        return []

    cx, cy, w, h = boxes_xywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 去 letterbox，回到原图
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)

    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms(boxes[idx], scores[idx], iou_thres)
        for k in keep:
            i = idx[k]
            dets.append((int(c), float(scores[i]), boxes[i].tolist()))
    dets.sort(key=lambda x: x[1], reverse=True)
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/yolov8n_int8.rknn")
    ap.add_argument("--image", default="assets/bus.jpg")
    ap.add_argument("--out", default="runs/bus_det.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"读图失败: {args.image}")
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb, scale, pad_x, pad_y = letterbox(img_rgb, 640)
    inp = np.expand_dims(lb, 0)  # NHWC uint8

    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        raise SystemExit("load_rknn failed")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime failed")

    # 预热
    rknn.inference(inputs=[inp])
    t0 = time.perf_counter()
    outs = rknn.inference(inputs=[inp])
    t1 = time.perf_counter()
    rknn.release()

    dets = postprocess(outs[0], scale, pad_x, pad_y, orig_w, orig_h, args.conf, args.iou)
    print(f"infer_ms={((t1 - t0) * 1000):.2f}  dets={len(dets)}")
    for cid, score, box in dets[:20]:
        name = COCO80[cid] if cid < len(COCO80) else str(cid)
        x1, y1, x2, y2 = [int(v) for v in box]
        print(f"  {name:16s} {score:.3f}  [{x1},{y1},{x2},{y2}]")
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            f"{name} {score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img_bgr)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
