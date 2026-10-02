#!/usr/bin/env python3
"""
Ground-truth mask check for instance segmentation.

Compares the C++ masks against ultralytics' own masks (not against
tools/ref_onnx.py, whose mask decode is still suspect) so we can tell whose
mask path is wrong.

Coordinate spaces
  cpp mask : (orig_h, orig_w)                original image
  ul  mask : (round(orig_h*s), round(orig_w*s))  content-cropped, scaled by s
             (s = letterbox scale = 640/1080)
so one pixel of the ul mask covers 1/s pixels of original space, and
    y_ul = Y_orig * s,  x_ul = X_orig * s
"""
import argparse
import json

import numpy as np


def box_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def mask_iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def bbox(m):
    yy, xx = np.where(m)
    if len(yy) == 0:
        return None
    return (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()))


def to_orig(ul_mask, s, oh, ow):
    """Resample an ultralytics (content-cropped) mask into original space."""
    h, w = ul_mask.shape
    yi = np.clip((np.arange(oh) * s).astype(int), 0, h - 1)
    xi = np.clip((np.arange(ow) * s).astype(int), 0, w - 1)
    return ul_mask[np.ix_(yi, xi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpp', required=True)
    ap.add_argument('--ref', required=True)
    args = ap.parse_args()

    cj = json.load(open(args.cpp))
    rj = json.load(open(args.ref))

    info = rj['meta']['letterbox']
    s = float(info['scale'])

    cpp_masks = np.asarray(cj['masks']) > 0.5
    ul_boxes = np.asarray(rj['ultralytics']['boxes'], dtype=float)
    ul_masks = np.asarray(rj['ultralytics']['masks']) > 0.5

    oh, ow = cpp_masks.shape[1], cpp_masks.shape[2]
    print(f'cpp masks {cpp_masks.shape}   ultralytics masks {ul_masks.shape}')
    print(f'letterbox scale={s:.5f} pad=({info["pad_left"]},{info["pad_top"]})\n')

    print(f'{"#":<3} {"cpp box":<26} {"ul box":<26} {"boxIoU":>7} '
          f'{"maskIoU":>8} {"areaC":>8} {"areaU":>8}')
    mious = []
    for i, cb in enumerate(cj['boxes']):
        cbox = [cb['x1'], cb['y1'], cb['x2'], cb['y2']]
        best_j, best_iou = -1, 0.0
        for j, ub in enumerate(ul_boxes):
            v = box_iou(cbox, ub)
            if v > best_iou:
                best_iou, best_j = v, j

        if best_j < 0:
            print(f'{i:<3} no match')
            continue

        um = to_orig(ul_masks[best_j], s, oh, ow)
        mi = mask_iou(cpp_masks[i], um)
        mious.append(mi)
        print(f'{i:<3} {str([round(v) for v in cbox]):<26} '
              f'{str([round(v) for v in ul_boxes[best_j]]):<26} '
              f'{best_iou:7.4f} {mi:8.4f} '
              f'{cpp_masks[i].sum():8d} {um.sum():8d}')

    if mious:
        print(f'\nmaskIoU mean={np.mean(mious):.4f} min={np.min(mious):.4f}')
        for i in range(min(2, len(mious))):
            print(f'  mask{i}: cpp bbox={bbox(cpp_masks[i])}')

    for i in range(min(2, len(ul_masks))):
        print(f'  ul mask{i} bbox(orig space)='
              f'{bbox(to_orig(ul_masks[i], s, oh, ow))}')


if __name__ == '__main__':
    main()