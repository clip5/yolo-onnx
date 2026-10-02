#!/usr/bin/env python3
"""
Compare yolo_onnx (C++) output against the python reference (tools/ref_onnx.py).

Matching is class-aware + IoU-based (greedy by score, like a real evaluator),
not index-by-index, so a reordering or a single extra low-score detection does
not read as "everything is wrong".

Usage:
  compare.py --cpp out/cpp/x.json --ref out/ref/x.json
  compare.py --glob 'out/cpp/*.json'        # pairs by basename with out/ref/
"""
import argparse
import glob
import json
import os

import numpy as np


def load_boxes(obj, key='boxes'):
    """Normalize boxes to (N,4) xyxy + scores + labels.

    The C++ dumper writes objects ({x1,y1,x2,y2,score,label}); the python
    reference writes parallel arrays (boxes/scores/labels). Accept both.
    """
    raw = obj.get(key)
    if raw is None:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)

    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        b = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in raw], dtype=float)
        s = np.array([d.get('score', 0.0) for d in raw], dtype=float)
        l = np.array([d.get('label', -1) for d in raw], dtype=int)
        return b.reshape(-1, 4), s, l

    b = np.asarray(raw, dtype=float).reshape(-1, 4)
    s = np.asarray(obj.get('scores', []), dtype=float).reshape(-1)
    l = np.asarray(obj.get('labels', []), dtype=int).reshape(-1)
    return b, s, l


def iou_matrix(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    union = aa[:, None] + ab[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match(cpp, ref, iou_thr=0.5):
    """Greedy score-ordered matching, class-aware. Returns list of (i, j, iou)."""
    cb, cs, cl = cpp
    rb, rs, rl = ref
    if cb.shape[0] == 0 or rb.shape[0] == 0:
        return []
    M = iou_matrix(cb, rb)
    pairs = []
    used_r = set()
    order = np.argsort(-cs)
    for i in order:
        best_j, best_iou = -1, iou_thr
        for j in range(rb.shape[0]):
            if j in used_r or cl[i] != rl[j]:
                continue
            if M[i, j] >= best_iou:
                best_iou, best_j = M[i, j], j
        if best_j >= 0:
            used_r.add(best_j)
            pairs.append((int(i), int(best_j), float(best_iou)))
    return pairs


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def mask_iou(a, b):
    """a, b: 2-D binary arrays -> IoU."""
    a = (np.asarray(a) > 0.5)
    b = (np.asarray(b) > 0.5)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def remap_mask_to_orig(mask, info, out_h, out_w):
    """Resample a mask given in model-input (letterboxed) space into original
    image space, so it can be compared against the C++ masks, which are
    produced at the original image size.

    Uses nearest-neighbour sampling on the same scale/pad the letterbox used.
    """
    m = np.asarray(mask, dtype=float)
    if m.shape[0] == out_h and m.shape[1] == out_w:
        return m
    scale = float(info['scale'])
    pad_l, pad_t = int(info['pad_left']), int(info['pad_top'])
    ys = np.clip((np.arange(out_h) * scale + pad_t).astype(int), 0, m.shape[0] - 1)
    xs = np.clip((np.arange(out_w) * scale + pad_l).astype(int), 0, m.shape[1] - 1)
    return m[np.ix_(ys, xs)]


def compare(cpp_path, ref_path, iou_thr=0.5, verbose=False):
    with open(cpp_path) as f:
        cj = json.load(f)
    with open(ref_path) as f:
        rj = json.load(f)

    cb, cs, cl = load_boxes(cj)
    rb, rs, rl = load_boxes(rj)

    pairs = match((cb, cs, cl), (rb, rs, rl), iou_thr)
    matched_c = {i for i, _, _ in pairs}

    out = dict(cpp=cpp_path, ref=ref_path,
               n_cpp=len(cb), n_ref=len(rb), n_matched=len(pairs))

    if pairs:
        ious = [p[2] for p in pairs]
        dx, dy = [], []
        ds = []
        for i, j, _ in pairs:
            dx.append(abs(cb[i, 0] - rb[j, 0]))
            dy.append(abs(cb[i, 1] - rb[j, 1]))
            ds.append(abs(cs[i] - rs[j]))
        out['iou_min'] = round(float(np.min(ious)), 4)
        out['iou_mean'] = round(float(np.mean(ious)), 4)
        out['coord_max_abs_diff'] = round(float(max(max(dx), max(dy))), 3)
        out['score_max_abs_diff'] = round(float(max(ds)), 4)
    else:
        out['iou_min'] = out['iou_mean'] = out['coord_max_abs_diff'] = 0.0
        out['score_max_abs_diff'] = 0.0

    out['missed_in_cpp'] = int(len(rb) - len(pairs))
    out['extra_in_cpp'] = int(len(cb) - len(pairs))

    # per-class label agreement (a wrong label shifts the whole box match)
    if len(pairs):
        out['label_mismatches'] = int(sum(1 for i, j, _ in pairs if cl[i] != rl[j]))
    else:
        out['label_mismatches'] = 0

    # masks (segment): reference masks live in model-input (letterboxed) space,
    # C++ masks are emitted at original image size — remap before comparing.
    cm, rm = cj.get('masks'), rj.get('masks')
    if cm is not None and rm is not None and len(cm) and len(rm):
        info = rj.get('meta', {}).get('letterbox')
        oh, ow = cj.get('image_height', 0), cj.get('image_width', 0)
        mious = []
        for i, j, _ in pairs:
            if i < len(cm) and j < len(rm):
                m = rm[j]
                if info and oh and ow:
                    m = remap_mask_to_orig(m, info, oh, ow)
                mious.append(mask_iou(cm[i], m))
        if mious:
            out['mask_iou_min'] = round(float(np.min(mious)), 4)
            out['mask_iou_mean'] = round(float(np.mean(mious)), 4)

    ul = rj.get('ultralytics')
    if ul is not None:
        ub = np.asarray(ul.get('boxes', []), dtype=float).reshape(-1, 4)
        out['n_ultralytics'] = len(ub)

    out['status'] = 'PASS' if (out['missed_in_cpp'] == 0
                               and out['iou_min'] >= iou_thr) else 'FAIL'
    return out, cj, rj


def fmt(o):
    s = (f"{os.path.basename(o['cpp']):42s} cpp={o['n_cpp']:3d} ref={o['n_ref']:3d} "
         f"match={o['n_matched']:3d} iou[min/mean]={o['iou_min']:.3f}/{o['iou_mean']:.3f} "
         f"dxy={o['coord_max_abs_diff']:.2f} dscore={o['score_max_abs_diff']:.4f} "
         f"miss={o['missed_in_cpp']} extra={o['extra_in_cpp']} lblbad={o['label_mismatches']}")
    if 'mask_iou_mean' in o:
        s += f" maskIoU={o['mask_iou_mean']:.4f}/{o['mask_iou_min']:.4f}"
    s += f"  [{o['status']}]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpp')
    ap.add_argument('--ref')
    ap.add_argument('--glob')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    pairs = []
    if args.glob:
        for c in sorted(glob.glob(args.glob)):
            base = os.path.basename(c)
            r = os.path.join('out/ref', base)
            if os.path.exists(r):
                pairs.append((c, r))
    else:
        pairs.append((args.cpp, args.ref))

    if not pairs:
        print('no pairs to compare')
        return

    n_pass = 0
    for c, r in pairs:
        try:
            o, cj, rj = compare(c, r, args.iou, args.verbose)
        except Exception as e:
            print(f'{os.path.basename(c):42s} ERROR {e!r}')
            continue
        print(fmt(o))
        n_pass += o['status'] == 'PASS'

        if args.verbose and o['status'] == 'FAIL':
            cb, cs, cl = load_boxes(cj)
            rb, rs, rl = load_boxes(rj)
            print('   --- ref boxes ---')
            for k in range(min(len(rb), 10)):
                print(f'     cls={rl[k]:3d} conf={rs[k]:.3f} {np.round(rb[k],1)}')
            print('   --- cpp boxes ---')
            for k in range(min(len(cb), 10)):
                print(f'     cls={cl[k]:3d} conf={cs[k]:.3f} {np.round(cb[k],1)}')

    print(f'\n{n_pass}/{len(pairs)} passed')


if __name__ == '__main__':
    main()