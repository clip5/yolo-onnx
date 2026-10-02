#!/usr/bin/env python3
"""
Definitive mask ground truth: run ultralytics with retina_masks=True so masks
come back at the ORIGINAL image resolution, then compare with the C++ masks
directly — no coordinate guesswork on our side.

Usage:
  check_masks_ul.py --model M.onnx --image IMG --cpp out/cpp/xxx.json
"""
import argparse
import json

import numpy as np
from ultralytics import YOLO


def box_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def bbox(m):
    yy, xx = np.where(m)
    return None if len(yy) == 0 else (int(xx.min()), int(yy.min()),
                                      int(xx.max()), int(yy.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--cpp', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    args = ap.parse_args()

    res = YOLO(args.model, task='segment').predict(
        args.image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        retina_masks=True, device='cpu', verbose=False)[0]

    ul_masks = (res.masks.data.cpu().numpy() > 0.5)
    ul_boxes = res.boxes.xyxy.cpu().numpy()
    ul_cls = res.boxes.cls.cpu().numpy().astype(int)
    print(f'ultralytics: {len(ul_masks)} masks {ul_masks.shape} (original space)')

    cj = json.load(open(args.cpp))
    cpp_masks = np.asarray(cj['masks']) > 0.5
    print(f'cpp:         {len(cpp_masks)} masks {cpp_masks.shape}')

    print(f'\n{"#":<3} {"cpp box":<24} {"ul box":<24} {"lbl":<6} {"boxIoU":>7} '
          f'{"maskIoU":>8} {"areaC":>8} {"areaU":>8}')
    mious = []
    for i, cb in enumerate(cj['boxes']):
        cbox = [cb['x1'], cb['y1'], cb['x2'], cb['y2']]
        best_j, best_iou, best_lbl = -1, 0.0, -1
        for j, ub in enumerate(ul_boxes):
            v = box_iou(cbox, ub)
            if v > best_iou:
                best_iou, best_j, best_lbl = v, j, int(ul_cls[j])
        um = ul_masks[best_j]
        mi = (cpp_masks[i] & um).sum() / max((cpp_masks[i] | um).sum(), 1)
        mious.append(mi)
        print(f'{i:<3} {str([round(v) for v in cbox]):<24} '
              f'{str([round(v) for v in ul_boxes[best_j]]):<24} '
              f'{cb["label"]}/{best_lbl:<4} {best_iou:7.4f} {mi:8.4f} '
              f'{cpp_masks[i].sum():8d} {um.sum():8d}')
        print(f'      cpp bbox={bbox(cpp_masks[i])}   ul bbox={bbox(um)}')

    if mious:
        print(f'\nmaskIoU mean={np.mean(mious):.4f} min={np.min(mious):.4f}')


if __name__ == '__main__':
    main()