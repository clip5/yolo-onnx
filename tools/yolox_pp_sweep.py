#!/usr/bin/env python3
"""
YOLOX preprocessing sweep.

Motivation: the decoder-included `assets/models/yolox_s.onnx` misses the bus on
bus.jpg. Two candidate causes: (a) the weights, (b) the preprocessing. This
sweeps the combinations that matter and scores each against a known-good
detector (yolo26s) instead of eyeballing.

Authoritative reference is the official YOLOX ONNX demo
(`YOLOX/yolox/data/data_augment.py::preproc`), which does:

    padded = full(input_h, input_w, 114)
    r = min(input_h/H, input_w/W)
    padded[:int(H*r), :int(W*r)] = resize(img, (int(W*r), int(H*r)))   # TOP-LEFT
    blob = transpose(padded, (2, 0, 1))                                # RGB, HWC->CHW
    blob = ascontiguousarray(blob, float32)                            # NO /255, NO mean/std

Two things the earlier sweep missed:
  * padding is TOP-LEFT (not centred), so the scale-back is a plain divide by r
  * NO normalisation at all — raw 0-255 float
"""
import argparse
import itertools
import json

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox(img, tw, th, align):
    """align='center' -> pad both sides; align='topleft' -> pad right/bottom."""
    h, w = img.shape[:2]
    r = min(tw / w, th / h)
    nw, nh = int(w * r), int(h * r)          # official uses int(), not round()
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, np.uint8)
    if align == 'topleft':
        canvas[:nh, :nw] = resized
        pad_l = pad_t = 0
    else:
        pad_l, pad_t = (tw - nw) // 2, (th - nh) // 2
        canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, r, pad_l, pad_t


def to_blob(canvas, swap_rb, scale, mean, std):
    b, g, r = cv2.split(canvas)
    ch = [r, g, b] if swap_rb else [b, g, r]
    out = [c.astype(np.float32) * (scale / std[i]) - mean[i] / std[i]
           for i, c in enumerate(ch)]
    return np.ascontiguousarray(np.stack(out, 0)[None, ...])


def decode(out, num_classes=80, conf=0.25):
    """cols 0:4 already pixel xywh (decoder-included export); col4/5+ sigmoided."""
    p = out[0]
    obj = p[:, 4]
    cls = p[:, 5:5 + num_classes]
    confs = obj * cls.max(1)
    labels = cls.argmax(1)
    sel = confs >= conf
    return p[sel, :4], confs[sel], labels[sel]


def to_orig(boxes, r, pad_l, pad_t, align):
    if align == 'topleft':
        return boxes / r
    out = boxes.copy()
    out[:, 0] = (out[:, 0] - pad_l) / r
    out[:, 1] = (out[:, 1] - pad_t) / r
    out[:, 2] = (out[:, 2] - pad_l) / r
    out[:, 3] = (out[:, 3] - pad_t) / r
    return out


def xyxy(wh):
    out = np.empty_like(wh)
    out[:, 0] = wh[:, 0] - wh[:, 2] / 2
    out[:, 1] = wh[:, 1] - wh[:, 3] / 2
    out[:, 2] = wh[:, 0] + wh[:, 2] / 2
    out[:, 3] = wh[:, 1] + wh[:, 3] / 2
    return out


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    it = ix * iy
    return it / ((a[2] - a[0]) * (a[3] - a[1]) +
                 (b[2] - b[0]) * (b[3] - b[1]) - it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='assets/models/yolox_s.onnx')
    ap.add_argument('--image', default='assets/images/bus.jpg')
    ap.add_argument('--ground-truth', default='output/cpp/yolo26s__bus__640.json',
                    help='known-good detector JSON, for scoring the sweep')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--size', type=int, default=640)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.model, so, providers=['CPUExecutionProvider'])

    gt = json.load(open(args.ground_truth))['boxes']
    gt_boxes = [(b['x1'], b['y1'], b['x2'], b['y2'], b['label']) for b in gt]
    print(f'ground truth ({args.ground_truth}): {len(gt_boxes)} boxes')
    for b in gt_boxes:
        print(f'   lbl={b[4]} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]')
    print()

    # (scale, mean, std) combos
    norms = {
        'raw0-255 (官方, 无归一化)': (1.0, (0., 0., 0.), (1., 1., 1.)),
        '/255 无归一化': (1 / 255., (0., 0., 0.), (1., 1., 1.)),
        '/255 + ImageNet (当前实现)': (1 / 255., IMAGENET_MEAN, IMAGENET_STD),
        'raw0-255 + ImageNet': (1.0, IMAGENET_MEAN, IMAGENET_STD),
    }

    rows = []
    for align, swap, (nname, (scale, mean, std)) in itertools.product(
            ('topleft', 'center'), (True, False), norms.items()):
        canvas, r, pl, pt = letterbox(img, args.size, args.size, align)
        blob = to_blob(canvas, swap, scale, mean, std)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        wh, confs, labels = decode(out, conf=args.conf)
        ob = xyxy(to_orig(wh, r, pl, pt, align)) if len(wh) else wh

        matched, ious = set(), []
        for g in gt_boxes:
            gb, glbl = g[0:4], g[4]
            best, bj = 0.0, -1
            for j, (b, lbl) in enumerate(zip(ob, labels)):
                if lbl != glbl or j in matched:
                    continue
                v = iou(gb, b)
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.5:
                matched.add(bj)
                ious.append(best)

        # also: is the bus detected at all?
        bus_hits = sum(1 for j, lbl in enumerate(labels)
                       if lbl == 5 and iou(gt_boxes[1][0:4], ob[j]) > 0.3) \
            if len(gt_boxes) > 1 else 0

        rows.append(dict(align=align, swap=swap, norm=nname,
                         n=len(ob), confmax=float(confs.max()) if len(confs) else 0.0,
                         matched=len(matched), n_gt=len(gt_boxes),
                         iou_mean=float(np.mean(ious)) if ious else 0.0,
                         iou_min=float(np.min(ious)) if ious else 0.0,
                         bus=bus_hits))

    rows.sort(key=lambda d: (-d['matched'], -d['iou_mean']))
    print('%-8s %-6s %-26s %4s %8s %8s %9s %5s' %
          ('align', 'swap', 'normalisation', 'n', 'confmax', 'matched',
           'iou_mean', 'bus'))
    print('-' * 84)
    for d in rows:
        star = '  <== 官方默认' if (d['align'] == 'topleft' and d['swap']
                                    and d['norm'].startswith('raw0-255 (')) else ''
        print('%-8s %-6s %-26s %4d %8.4f %5d/%-2d %9.3f %5d%s' %
              (d['align'], 'RGB' if d['swap'] else 'BGR', d['norm'], d['n'],
               d['confmax'], d['matched'], d['n_gt'], d['iou_mean'],
               d['bus'], star))


if __name__ == '__main__':
    main()