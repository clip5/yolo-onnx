#!/usr/bin/env python3
"""
Generate a human-readable accuracy report + annotated comparison images from
the JSON files produced by tools/dump_json and tools/ref_onnx.py.

Outputs (under --out, default ./output):
  REPORT.md          summary tables: per-file box comparison + numbers
  vis/*.jpg          side-by-side annotated images: C++ (left) vs reference (right)
  vis/*_seg.jpg      instance masks overlaid (segment only)
  boxes/*.txt        plain-text box list per file (no JSON digging needed)

Usage:
  report.py --cpp-dir output/cpp --ref-dir output/ref
"""
import argparse
import json
import os

import cv2
import numpy as np

COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def load_boxes(obj):
    raw = obj.get('boxes')
    if raw is None:
        return []
    if raw and isinstance(raw[0], dict):
        return [(b['x1'], b['y1'], b['x2'], b['y2'], b['score'], int(b['label']))
                for b in raw]
    # 旧版 ref JSON 里 label 可能被写成 float（0.0），统一转 int
    return [(b[0], b[1], b[2], b[3], s, int(l)) for b, s, l in
            zip(obj['boxes'], obj['scores'], obj['labels'])]


def class_name(lbl):
    return COCO[lbl] if 0 <= lbl < len(COCO) else str(lbl)


def draw(img, boxes, title, color):
    vis = img.copy()
    for (x1, y1, x2, y2, s, l) in boxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, color, 2)
        text = f'{class_name(l)} {s:.2f}'
        # keep the label inside the frame when the box sits at the top edge
        ty = max(p1[1] - 6, 12)
        cv2.putText(vis, text, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(vis, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def side_by_side(img, cpp_boxes, ref_boxes, title):
    h, w = img.shape[:2]
    gap, head = 8, 30
    canvas = np.full((h + head, w * 2 + gap, 3), 40, np.uint8)
    left = draw(img, cpp_boxes, f'{title}  |  yolo_onnx (C++)', (0, 230, 0))
    right = draw(img, ref_boxes, 'reference (onnxruntime decode)', (0, 140, 255))
    canvas[head:, :w] = left
    canvas[head:, w + gap:] = right
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def mask_overlay(img, cpp_masks, boxes, title):
    """Green overlay of the C++ instance masks, boxes on top."""
    vis = img.copy()
    colors = [(0, 255, 0), (0, 200, 255), (255, 128, 0),
              (255, 0, 255), (0, 128, 255)]
    for i, m in enumerate(cpp_masks):
        mm = (np.asarray(m) > 0.5)
        if mm.shape[:2] != img.shape[:2]:
            continue
        col = np.array(colors[i % len(colors)], np.uint8)
        vis[mm] = (vis[mm] * 0.5 + col * 0.5).astype(np.uint8)
    for i, (x1, y1, x2, y2, s, l) in enumerate(boxes):
        col = tuple(int(c) for c in colors[i % len(colors)])
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(vis, f'{class_name(l)} {s:.2f}', (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(vis, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpp-dir', default='output/cpp')
    ap.add_argument('--ref-dir', default='output/ref')
    ap.add_argument('--images', default='assets/images')
    ap.add_argument('--out', default='output')
    args = ap.parse_args()

    vis_dir = os.path.join(args.out, 'vis')
    txt_dir = os.path.join(args.out, 'boxes')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compare import compare  # reuse the same matching logic

    names = sorted(f for f in os.listdir(args.cpp_dir) if f.endswith('.json'))
    rows, img_cache, vis_n, seg_n = [], {}, 0, 0

    for name in names:
        cpp_path = os.path.join(args.cpp_dir, name)
        ref_path = os.path.join(args.ref_dir, name)
        if not os.path.exists(ref_path):
            continue

        cj = json.load(open(cpp_path))
        rj = json.load(open(ref_path))
        base = name[:-5]
        cpp_boxes = load_boxes(cj)
        ref_boxes = load_boxes(rj)

        # ---- summary row ----
        m = compare(cpp_path, ref_path, iou_thr=0.5)[0]
        rows.append((base, m, cj, rj))

        # ---- plain-text box list ----
        lines = [f'{base}   ({cj.get("task")}, input '
                 f'{cj.get("input_width")}x{cj.get("input_height")}, '
                 f'image {cj.get("image_width")}x{cj.get("image_height")})',
                 f'{"cls":<22} {"score":>7}   {"x1":>7} {"y1":>7} '
                 f'{"x2":>7} {"y2":>7}   (C++ / original image space)', '-' * 78]
        for (x1, y1, x2, y2, s, l) in cpp_boxes:
            lines.append(f'{class_name(l):<22} {s:7.4f}   {x1:7.1f} {y1:7.1f} '
                         f'{x2:7.1f} {y2:7.1f}')
        with open(os.path.join(txt_dir, base + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

        # ---- annotated image ----
        stem = base.split('__')
        img_name = stem[1] + '.jpg' if len(stem) > 1 else None
        size = stem[2] if len(stem) > 2 else ''
        if not img_name or not os.path.exists(os.path.join(args.images, img_name)):
            continue
        if img_name not in img_cache:
            img_cache[img_name] = cv2.imread(os.path.join(args.images, img_name))
        img = img_cache[img_name]
        if img is None:
            continue

        title = f'{stem[0]}  {img_name.replace(".jpg", "")}  @ {size}'
        cv2.imwrite(os.path.join(vis_dir, base + '.jpg'),
                    side_by_side(img, cpp_boxes, ref_boxes, title))
        vis_n += 1

        if cj.get('masks'):
            cv2.imwrite(os.path.join(vis_dir, base + '_seg.jpg'),
                        mask_overlay(img, cj['masks'], cpp_boxes,
                                     title + '   [instance masks]'))
            seg_n += 1

    # ---------------- report ----------------
    md = ['# yolo-onnx 精度核对报告', '',
          'C++ (`yolo_onnx`) vs 参考实现（同一 ONNX，onnxruntime 显式解码）。',
          '匹配方式：按类别 + IoU 贪心匹配（不是按下标），故顺序不同不影响结论。', '',
          '**PASS** = 参考实现的每个框都被匹配上且 IoU ≥ 0.5。', '',
          '## 总览', '',
          '| 用例 | 任务 | 输入尺寸 | C++框 | 参考框 | 匹配 | IoU min/mean | 坐标最大偏差 | 分数最大偏差 | 结论 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    npass = 0
    for base, m, cj, _ in rows:
        npass += m['status'] == 'PASS'
        md.append(f'| `{base}` | {cj.get("task")} | '
                  f'{cj.get("input_width")}x{cj.get("input_height")} | '
                  f'{m["n_cpp"]} | {m["n_ref"]} | {m["n_matched"]} | '
                  f'{m["iou_min"]:.3f} / {m["iou_mean"]:.3f} | '
                  f'{m["coord_max_abs_diff"]:.2f}px | '
                  f'{m["score_max_abs_diff"]:.4f} | '
                  f'{"✅ PASS" if m["status"] == "PASS" else "❌ FAIL"} |')

    md += ['', f'**合计 {npass}/{len(rows)} 通过**', '']

    md += ['## 逐用例框明细（C++ 输出，原图坐标）', '']
    for base, m, cj, _ in rows:
        boxes = load_boxes(cj)
        md.append(f'### `{base}` — {cj.get("task")}, '
                  f'输入 {cj.get("input_width")}x{cj.get("input_height")}, '
                  f'匹配 {m["n_matched"]}/{m["n_ref"]}, IoU min {m["iou_min"]:.3f}')
        md.append('')
        md.append('| 类别 | 分数 | x1 | y1 | x2 | y2 |')
        md.append('|---|---|---|---|---|---|')
        for (x1, y1, x2, y2, s, l) in boxes:
            md.append(f'| {class_name(l)} | {s:.4f} | {x1:.1f} | {y1:.1f} | '
                      f'{x2:.1f} | {y2:.1f} |')
        md.append('')

    md += ['## 可视化文件', '',
           f'- `output/vis/*.jpg`（{vis_n} 张）：左= C++ 绿色框，右 = 参考实现橙色框。',
           '  两者重合即为对齐；可直观看出是否存在漏检/多检/框偏移。',
           f'- `output/vis/*_seg.jpg`（{seg_n} 张）：实例掩码叠加（C++ 生成的 mask）。',
           f'- `output/boxes/*.txt`（{len(rows)} 份）：每个用例的纯文本框列表。', '']

    with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(md))

    print(f'report: {args.out}/REPORT.md  ({npass}/{len(rows)} PASS)')
    print(f'images: {vis_dir}  ({vis_n} compare + {seg_n} mask)')
    print(f'boxes : {txt_dir}  ({len(rows)} txt)')


if __name__ == '__main__':
    main()