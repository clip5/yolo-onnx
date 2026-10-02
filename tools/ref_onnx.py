#!/usr/bin/env python3
"""
Reference generator: run an ONNX model with onnxruntime and decode boxes/masks
with the *upstream* reference logic (ultralytics ops where possible, otherwise
an explicit re-implementation), then dump JSON for comparison against the C++
yolo_onnx output.

Two decode paths are emitted when both are available:
  ref_ort_*  : hand-written decode of the raw ONNX output (mirrors what a
               correct decoder must do: letterbox -> blob -> decode -> NMS)
  ref_ul_*   : ultralytics' own predictor on the same .onnx (black box,
               applies its own letterbox/NMS/threshold) -- used to judge
               whether the C++ pipeline is in the right ballpark

Usage:
  ref_onnx.py --model assets/models/yolo11n.onnx --task detect \
              --image assets/images/bus.jpg --size 640 --conf 0.25 --iou 0.45 \
              --out out/ref/yolo11n__bus__640.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort


# --------------------------------------------------------------- preprocess
def letterbox(img, new_shape, color=(114, 114, 114), swap=True, scale_fill=False):
    """Ultralytics LetterBox: keep aspect, pad, centered.

    Returns (canvas, info) with info in the same spirit as LetterboxInfo:
      scale, pad_left, pad_top, orig_w, orig_h, target_w, target_h
    """
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        th, tw = new_shape, new_shape
    else:
        th, tw = new_shape[0], new_shape[1]

    r = min(tw / w, th / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if scale_fill:
        new_w, new_h = tw, th
        r = max(tw / w, th / h)
        dw, dh = tw - new_w, th - new_h
        canvas = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        top, left = dh // 2, dw // 2
        canvas = canvas[top:top + new_h, left:left + new_w]
        pad = (0, 0)
    else:
        dw, dh = tw - new_w, th - new_h
        canvas = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, left = dh // 2, dw // 2

    top = max(0, int(round(top)))
    left = max(0, int(round(left)))
    if not scale_fill:
        padded = cv2.copyMakeBorder(canvas, top, dh - top, left, dw - left,
                                    cv2.BORDER_CONSTANT, value=color)
        canvas = padded
    else:
        canvas = cv2.copyMakeBorder(canvas, top, dh - top, left, dw - left,
                                    cv2.BORDER_CONSTANT, value=color)

    info = dict(scale=r, pad_left=left, pad_top=top,
                orig_w=w, orig_h=h, target_w=tw, target_h=th)
    return canvas, info


def make_blob(canvas_bgr, swap_rb=True, scale=1.0 / 255.0, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """canvas (BGR uint8) -> NCHW float32."""
    b, g, r = cv2.split(canvas_bgr)
    chans = [r, g, b] if swap_rb else [b, g, r]
    out = []
    for i, c in enumerate(chans):
        a = scale / std[i]
        bb = -mean[i] / std[i]
        out.append(c.astype(np.float32) * a + bb)
    blob = np.stack(out, axis=0)[None, ...]
    return np.ascontiguousarray(blob)


# ------------------------------------------------------------------- NMS ops
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


def nms(boxes, scores, iou_thresh):
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def nms_class_aware(boxes, scores, labels, iou_thresh):
    """Per-class NMS. Returns GLOBAL row indices (into the original arrays)."""
    keep_all = []
    for c in np.unique(labels):
        rows = np.where(labels == c)[0]
        keep_all.extend(rows[nms(boxes[rows], scores[rows], iou_thresh)].tolist())
    return keep_all


def finalize(boxes_xywh, scores, labels, info, iou, max_det=0):
    """xywh(model space) + conf/label -> class-aware NMS -> scale back."""
    if boxes_xywh.shape[0] == 0:
        return dict(boxes=np.zeros((0, 4), np.float32), scores=np.zeros((0,), np.float32),
                    labels=np.zeros((0,), int))
    b = xywh2xyxy(boxes_xywh.astype(np.float32))
    keep = np.array(nms_class_aware(b, scores, labels, iou), int)
    keep = keep[np.argsort(-scores[keep])]          # score-descending
    if max_det:
        keep = keep[:max_det]
    return dict(boxes=scale_boxes(b[keep], info), scores=scores[keep], labels=labels[keep])


def scale_boxes(boxes, info):
    out = boxes.copy()
    out[:, 0] = (out[:, 0] - info['pad_left']) / info['scale']
    out[:, 1] = (out[:, 1] - info['pad_top']) / info['scale']
    out[:, 2] = (out[:, 2] - info['pad_left']) / info['scale']
    out[:, 3] = (out[:, 3] - info['pad_top']) / info['scale']
    out[:, 0] = np.clip(out[:, 0], 0, info['orig_w'])
    out[:, 1] = np.clip(out[:, 1], 0, info['orig_h'])
    out[:, 2] = np.clip(out[:, 2], 0, info['orig_w'])
    out[:, 3] = np.clip(out[:, 3], 0, info['orig_h'])
    return out


# ---------------------------------------------------------- model metadata
V5_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
              [[30, 61], [62, 45], [59, 119]],
              [[116, 90], [156, 198], [373, 326]]]


def onnx_meta(path):
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        return {p.key: p.value for p in m.metadata_props}
    except Exception as e:  # pragma: no cover
        return {"_error": str(e)}


def n_classes_from_meta(path, default=80):
    md = onnx_meta(path)
    names = md.get('names')
    if not names:
        return default
    try:
        import ast
        d = ast.literal_eval(names)
        if isinstance(d, dict):
            return len(d)
    except Exception:
        pass
    return default


# --------------------------------------------------------------- decoders
def decode_v8_detect(out, num_classes, conf, iou, info):
    """v8/v11/v26 detect: [1, 4+C, N] xywh pixels + already-sigmoid cls."""
    p = out[0].T                      # N x C
    cls = p[:, 4:4 + num_classes]
    labels = cls.argmax(1)
    scores = cls.max(1)
    sel = scores >= conf
    return finalize(p[sel, :4], scores[sel], labels[sel], info, iou)


def decode_yolox_detect(out, num_classes, conf, iou, info, imgsz_w, imgsz_h):
    """yolox: [1, N, 5+C] — cols 0:4 are RAW box logits, col4 obj and
    5: class scores are ALREADY sigmoided (see the ONNX graph: the concat is
    conv2d_reg (raw) + sigmoid(head)). Boxes need the official decode:
        center = (sigmoid(t)*2 - 0.5 + grid) * stride
        size   = (sigmoid(t)*2)^2 * stride
    """
    p = out[0]
    obj = p[:, 4]
    cls = p[:, 5:5 + num_classes]
    labels = cls.argmax(1)
    scores = obj * cls.max(1)
    sel = scores >= conf

    boxes = []
    strides = [8, 16, 32]
    grids = [(imgsz_w // s, imgsz_h // s) for s in strides]
    offset = 0
    for level, stride in enumerate(strides):
        gw, gh = grids[level]
        count = gw * gh
        idxs = np.arange(offset, min(offset + count, p.shape[0]))
        if len(idxs) == 0:
            break
        g = idxs - offset
        gi, gj = g % gw, g // gw
        r = p[idxs]
        cx = (sigmoid(r[:, 0]) * 2 - 0.5 + gi) * stride
        cy = (sigmoid(r[:, 1]) * 2 - 0.5 + gj) * stride
        bw = (sigmoid(r[:, 2]) * 2) ** 2 * stride
        bh = (sigmoid(r[:, 3]) * 2) ** 2 * stride
        boxes.append(np.stack([cx, cy, bw, bh], axis=1))
        offset += count
        if offset >= p.shape[0]:
            break
    if not boxes:
        return finalize(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int), info, iou)
    allb = np.concatenate(boxes, axis=0)[:p.shape[0]]
    return finalize(allb[sel], scores[sel], labels[sel], info, iou)


def decode_v5_detect(out, num_classes, conf, iou, info):
    """v5: [1, N, 5+C] xywh pixels, col4 = obj (sigmoid), 5: = cls (sigmoid)."""
    p = out[0]
    obj = p[:, 4]
    cls = p[:, 5:5 + num_classes]
    labels = cls.argmax(1)
    scores = obj * cls.max(1)
    sel = scores >= conf
    return finalize(p[sel, :4], scores[sel], labels[sel], info, iou)


def crop_mask(masks, boxes):
    n, h, w = masks.shape
    out = np.zeros_like(masks)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(w, np.ceil(x2)))
        y2 = int(min(h, np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return out


def process_mask(protos, masks_in, bboxes, shape, upsample=True):
    c, mh, mw = protos.shape
    ih, iw = shape
    masks = sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    width_ratio = mw / iw
    height_ratio = mh / ih
    if upsample:
        masks = np.transpose(masks, (0, 2, 1))
        masks = np.array([cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR)
                          for m in masks])[:, None, :, :]
        masks = np.squeeze(masks, axis=1)
        masks = np.clip(masks, 0.0, 1.0)
        masks = crop_mask(masks, bboxes)
    return masks


def decode_v8_segment(out, num_classes, num_masks, conf, iou, info, imgsz):
    """v8/v11 segment: out0=[1,4+C+nm,N] (cls already sigmoided), out1=proto[1,nm,mh,mw].

    Boxes stay in model-input space for mask cropping; the returned boxes are
    scaled back to the original image.
    """
    p = out[0][0].T                   # N x (4+C+nm)
    proto = out[1][0]                  # nm x mh x mw

    cls = p[:, 4:4 + num_classes]
    labels = cls.argmax(1)
    scores = cls.max(1)
    sel = scores >= conf
    p, scores, labels = p[sel], scores[sel], labels[sel]

    b = xywh2xyxy(p[:, :4].astype(np.float32))
    keep = np.array(nms_class_aware(b, scores, labels, iou), int)
    keep = keep[np.argsort(-scores[keep])]
    p, scores, labels = p[keep], scores[keep], labels[keep]
    b = b[keep]

    coeffs = p[:, 4 + num_classes:4 + num_classes + num_masks]
    masks = process_mask(proto, coeffs, b, imgsz, upsample=True)
    return dict(boxes=scale_boxes(b, info), scores=scores, labels=labels, masks=masks)


def run_ort(path, task, img_bgr, imgsz, conf, iou, num_threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    model_type = detect_model_type(path, task)
    nc = n_classes_from_meta(path)

    if isinstance(imgsz, int):
        size = (imgsz, imgsz)   # (w, h)
    else:
        size = (imgsz[1], imgsz[0])

    swap_rb = model_type != 'ppyoloe'
    mean, std = (0., 0., 0.), (1., 1., 1.)
    if model_type == 'yolox':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    canvas, info = letterbox(img_bgr, (size[1], size[0]))
    blob = make_blob(canvas, swap_rb=swap_rb, mean=mean, std=std)
    inp = sess.get_inputs()[0]
    outs = sess.run(None, {inp.name: blob})

    meta = dict(model_type=model_type, task=task, num_classes=nc,
                input_shape=list(blob.shape), letterbox=info,
                onnx_outputs=[list(o.shape) for o in outs])

    if task == 'detect':
        if model_type == 'v5':
            d = decode_v5_detect(outs[0], nc, conf, iou, info)
        elif model_type == 'yolox':
            d = decode_yolox_detect(outs[0], nc, conf, iou, info, size[0], size[1])
        else:
            d = decode_v8_detect(outs[0], nc, conf, iou, info)
        # finalize() already scaled boxes back to original-image space
        return dict(meta=meta,
                    boxes=d['boxes'], scores=d['scores'], labels=d['labels'],
                    masks=None)

    if task == 'segment':
        num_masks = outs[1].shape[1]
        d = decode_v8_segment(outs, nc, num_masks, conf, iou, info,
                              (size[1], size[0]))
        return dict(meta=meta,
                    raw_meta=dict(input_shape=list(blob.shape),
                                  shapes=[list(o.shape) for o in outs]),
                    boxes=d['boxes'], scores=d['scores'], labels=d['labels'],
                    masks=d['masks'])
    raise ValueError(task)


def detect_model_type(path, task):
    md = onnx_meta(path)
    if 'task' in md:
        t = md['task']
        if 'yolox' in path.lower():
            return 'yolox'
        if 'v5' in path.lower() and 'yolov5' in md.get('description', '').lower():
            return 'v5'
        if 'YOLOv5' in md.get('description', '') or 'yolov5' in path.lower():
            return 'v5'
        if 'YOLOX' in md.get('description', ''):
            return 'yolox'
    if 'yolox' in path.lower():
        return 'yolox'
    if 'yolov5' in path.lower():
        return 'v5'
    return 'v8'


def run_ultralytics(path, img_path, imgsz, conf, iou, task):
    from ultralytics import YOLO
    m = YOLO(path, task=task)
    res = m.predict(img_path, imgsz=imgsz, conf=conf, iou=iou,
                    device='cpu', verbose=False)[0]
    out = dict()
    if res.boxes is not None and len(res.boxes):
        out['boxes'] = res.boxes.xyxy.cpu().numpy()
        out['scores'] = res.boxes.conf.cpu().numpy()
        out['labels'] = res.boxes.cls.cpu().numpy().astype(int)
    else:
        out['boxes'] = np.zeros((0, 4), np.float32)
        out['scores'] = np.zeros((0,), np.float32)
        out['labels'] = np.zeros((0,), int)
    if task == 'segment' and getattr(res, 'masks', None) is not None and len(res.masks.data):
        out['masks'] = res.masks.data.cpu().numpy()
        out['mask_shape'] = list(out['masks'].shape)
    else:
        out['masks'] = None
    return out


def to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.astype(float).tolist() if o.dtype != bool else o.astype(int).tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', default='detect', choices=['detect', 'segment'])
    ap.add_argument('--image', required=True)
    ap.add_argument('--size', default='640',
                    help='int or WxH, e.g. 640 or 640x384')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-ul', action='store_true', help='skip ultralytics path')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if 'x' in args.size:
        w, h = args.size.lower().split('x')
        imgsz = [int(h), int(w)]   # ultralytics imgsz = [h, w]
    else:
        imgsz = int(args.size)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit('cannot read image ' + args.image)

    ort_res = run_ort(args.model, args.task, img, imgsz if not isinstance(imgsz, list) else imgsz,
                      args.conf, args.iou, args.threads)

    result = dict(
        source='onnxruntime+explicit-decode',
        model=args.model, image=args.image,
        conf=args.conf, iou=args.iou,
        meta=ort_res['meta'],
        boxes=to_jsonable(ort_res['boxes']),
        scores=to_jsonable(ort_res['scores']),
        labels=to_jsonable(ort_res['labels']),
    )
    if ort_res['masks'] is not None:
        result['masks'] = to_jsonable(ort_res['masks'].astype(np.float32))
        result['mask_shape'] = [ort_res['masks'].shape[0],
                                ort_res['masks'].shape[1],
                                ort_res['masks'].shape[2]]

    if not args.no_ul:
        try:
            ul = run_ultralytics(args.model, args.image, imgsz,
                                 args.conf, args.iou, args.task)
            result['ultralytics'] = dict(
                boxes=to_jsonable(ul['boxes']),
                scores=to_jsonable(ul['scores']),
                labels=to_jsonable(ul['labels']),
            )
            if ul['masks'] is not None:
                result['ultralytics']['masks'] = to_jsonable(ul['masks'])
                result['ultralytics']['mask_shape'] = ul['mask_shape']
        except Exception as e:
            result['ultralytics_error'] = repr(e)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'wrote {args.out}: {len(result["boxes"])} boxes '
          f'(ultralytics: {len(result.get("ultralytics", {}).get("boxes", []))})')


if __name__ == '__main__':
    main()