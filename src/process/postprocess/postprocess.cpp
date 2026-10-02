#include "process/postprocess/postprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// IoU / NMS（普通框）
// ============================================================
float iou(const Box& a, const Box& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<int> nms(const BoxArray& boxes, float iou_threshold) {
    std::vector<int> indices;
    std::vector<int> order(boxes.size());
    for (int i = 0; i < (int)boxes.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return boxes[i].score > boxes[j].score;
    });
    std::vector<bool> removed(boxes.size(), false);
    for (int i = 0; i < (int)order.size(); i++) {
        if (removed[order[i]]) continue;
        indices.push_back(order[i]);
        for (int j = i + 1; j < (int)order.size(); j++) {
            if (removed[order[j]]) continue;
            if (iou(boxes[order[i]], boxes[order[j]]) > iou_threshold) {
                removed[order[j]] = true;
            }
        }
    }
    return indices;
}

// ============================================================
// OBB 旋转 IoU / NMS
// ============================================================
float obb_iou(const OBBBox& a, const OBBBox& b) {
    cv::RotatedRect rr_a(cv::Point2f(a.cx, a.cy), cv::Size2f(a.w, a.h), a.angle * 180.0f / CV_PI);
    cv::RotatedRect rr_b(cv::Point2f(b.cx, b.cy), cv::Size2f(b.w, b.h), b.angle * 180.0f / CV_PI);

    std::vector<cv::Point2f> inter;
    float intersection_area = 0.0f;
    int ret = cv::rotatedRectangleIntersection(rr_a, rr_b, inter);
    if (ret != 0 && !inter.empty()) {
        intersection_area = cv::contourArea(inter);
    }

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - intersection_area;
    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

std::vector<int> obb_nms(const OBBBoxArray& boxes, float iou_threshold) {
    std::vector<int> indices;
    std::vector<int> order(boxes.size());
    for (int i = 0; i < (int)boxes.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return boxes[i].score > boxes[j].score;
    });
    std::vector<bool> removed(boxes.size(), false);
    for (int i = 0; i < (int)order.size(); i++) {
        if (removed[order[i]]) continue;
        indices.push_back(order[i]);
        for (int j = i + 1; j < (int)order.size(); j++) {
            if (removed[order[j]]) continue;
            if (obb_iou(boxes[order[i]], boxes[order[j]]) > iou_threshold) {
                removed[order[j]] = true;
            }
        }
    }
    return indices;
}

// ============================================================
// 坐标还原（letterbox 逆映射）
// ============================================================
void PostProcess::restore_boxes(BoxArray& boxes, const LetterboxInfo& lb) {
    for (auto& box : boxes) scale_box(box, lb);
}

void PostProcess::restore_keypoints(std::vector<Keypoint>& kps, const LetterboxInfo& lb) {
    for (auto& kp : kps) scale_keypoint(kp, lb);
}

void PostProcess::restore_obb(OBBBoxArray& boxes, const LetterboxInfo& lb) {
    for (auto& box : boxes) scale_obb_box(box, lb);
}

// ============================================================
// 置信度过滤
// ============================================================
BoxArray PostProcess::filter_by_score(const BoxArray& boxes) const {
    if (params_.score_thresh <= 0.0f) return boxes;
    BoxArray out;
    out.reserve(boxes.size());
    for (const auto& b : boxes) {
        if (b.score >= params_.score_thresh) out.push_back(b);
    }
    return out;
}

OBBBoxArray PostProcess::filter_by_score(const OBBBoxArray& boxes) const {
    if (params_.score_thresh <= 0.0f) return boxes;
    OBBBoxArray out;
    out.reserve(boxes.size());
    for (const auto& b : boxes) {
        if (b.score >= params_.score_thresh) out.push_back(b);
    }
    return out;
}

// ============================================================
// NMS
// ============================================================
BoxArray PostProcess::suppress(const BoxArray& boxes) const {
    auto keep = nms(boxes, params_.nms_thresh);
    BoxArray out;
    out.reserve(keep.size());
    for (int idx : keep) out.push_back(boxes[idx]);
    return out;
}

OBBBoxArray PostProcess::suppress_rotated(const OBBBoxArray& boxes) const {
    auto keep = obb_nms(boxes, params_.nms_thresh);
    OBBBoxArray out;
    out.reserve(keep.size());
    for (int idx : keep) out.push_back(boxes[idx]);
    return out;
}

// ============================================================
// top_k
// ============================================================
BoxArray PostProcess::top_k(BoxArray boxes, int max_detections) {
    if (max_detections <= 0 || (int)boxes.size() <= max_detections) return boxes;
    std::partial_sort(boxes.begin(), boxes.begin() + max_detections, boxes.end(),
                      [](const Box& a, const Box& b) { return a.score > b.score; });
    boxes.resize(max_detections);
    return boxes;
}

OBBBoxArray PostProcess::top_k(OBBBoxArray boxes, int max_detections) {
    if (max_detections <= 0 || (int)boxes.size() <= max_detections) return boxes;
    std::partial_sort(boxes.begin(), boxes.begin() + max_detections, boxes.end(),
                      [](const OBBBox& a, const OBBBox& b) { return a.score > b.score; });
    boxes.resize(max_detections);
    return boxes;
}

// ============================================================
// select：过滤 → 还原 → NMS，返回保留的候选下标（按分数降序）
// ============================================================
// segment/pose 需要按 box 下标回取附属信息（mask 系数 / 关键点），
// 因此必须返回索引而不是框本身。
std::vector<int> PostProcess::select(const std::vector<Box>& candidates,
                                     const LetterboxInfo& lb) const {
    // 1. 置信度过滤（保留原始下标）
    std::vector<int> idx;
    idx.reserve(candidates.size());
    for (int i = 0; i < (int)candidates.size(); i++) {
        if (params_.score_thresh <= 0.0f || candidates[i].score >= params_.score_thresh) {
            idx.push_back(i);
        }
    }
    if (idx.empty()) return {};

    // 2. 还原到原图坐标系后再做 NMS（跨尺度的框不应互相抑制）
    BoxArray restored;
    restored.reserve(idx.size());
    for (int i : idx) {
        Box b = candidates[i];
        scale_box(b, lb);
        restored.push_back(b);
    }

    // 3. NMS，返回的是 restored 的下标（已按分数降序）
    auto kept = nms(restored, params_.nms_thresh);

    std::vector<int> keep;
    keep.reserve(kept.size());
    for (int k : kept) keep.push_back(idx[k]);

    // 4. 数量限制（keep 已是分数降序，可直接截断）
    if (params_.max_detections > 0 && (int)keep.size() > params_.max_detections) {
        keep.resize(params_.max_detections);
    }
    return keep;
}

// ============================================================
// PostProcessDet
// ============================================================
InferResult PostProcessDet::forward(const TensorSet& outputs,
                                    const LetterboxInfo& lb) const {
    BoxArray candidates = decoder_->decode_detect(outputs, ctx_);
    if (candidates.empty()) return DetectResult{};

    auto keep = select(candidates, lb);
    DetectResult result;
    result.boxes.reserve(keep.size());
    for (int i : keep) {
        // select() 返回的是 candidates 的下标（模型输入坐标系），
        // 必须再还原一次才能得到原图坐标
        Box b = candidates[i];
        scale_box(b, lb);
        result.boxes.push_back(b);
    }
    return result;
}

// ============================================================
// PostProcessPose
// ============================================================
InferResult PostProcessPose::forward(const TensorSet& outputs,
                                     const LetterboxInfo& lb) const {
    auto cands = decoder_->decode_pose(outputs, ctx_);
    if (cands.empty()) return PoseResult{};

    std::vector<Box> boxes;
    boxes.reserve(cands.size());
    for (const auto& c : cands) boxes.push_back(c.box);

    auto keep = select(boxes, lb);

    PoseResult result;
    result.boxes.reserve(keep.size());
    result.keypoints.reserve(keep.size());
    for (int i : keep) {
        Box b = cands[i].box;
        scale_box(b, lb);
        auto kps = cands[i].extra;
        restore_keypoints(kps, lb);
        result.boxes.push_back(b);
        result.keypoints.push_back(std::move(kps));
    }
    return result;
}

// ============================================================
// PostProcessOBB
// ============================================================
InferResult PostProcessOBB::forward(const TensorSet& outputs,
                                    const LetterboxInfo& lb) const {
    OBBBoxArray candidates = decoder_->decode_obb(outputs, ctx_);
    if (candidates.empty()) return OBBResult{};

    auto boxes = filter_by_score(candidates);
    restore_obb(boxes, lb);
    boxes = suppress_rotated(boxes);
    return OBBResult{top_k(std::move(boxes), params_.max_detections)};
}

// ============================================================
// PostProcessSegment
// ============================================================
namespace {
constexpr int kNumMasks = 32;   // mask 系数通道数
}

int PostProcessSegment::find_proto_index(const TensorSet& outputs) {
    for (size_t i = 0; i < outputs.size(); i++) {
        const auto& shape = outputs.shape(i);
        // proto mask: [1, 32, H, W]，H/W 通常为输入的 1/4
        if (shape.size() == 4 && shape[1] == kNumMasks && shape[2] > 0 && shape[3] > 0) {
            return (int)i;
        }
    }
    return -1;
}

std::vector<Mask> PostProcessSegment::compose_masks(
    const TensorSet& outputs, int proto_idx,
    const std::vector<std::array<float, kNumMasks>>& coeffs,
    const LetterboxInfo& lb,
    const std::vector<Box>& boxes,
    int input_width) {

    std::vector<Mask> masks;
    const auto& proto_shape = outputs.shape(proto_idx);
    const int proto_h = (int)proto_shape[2];
    const int proto_w = (int)proto_shape[3];
    const int orig_w  = lb.orig_w;
    const int orig_h  = lb.orig_h;
    if (proto_h <= 0 || proto_w <= 0) return masks;

    const float* proto = outputs.data(proto_idx);

    for (size_t k = 0; k < boxes.size() && k < coeffs.size(); k++) {
        const auto& coeff = coeffs[k];
        const Box& box    = boxes[k];

        // mask = sigmoid(proto @ coeff)，proto 为 [32, proto_h, proto_w]
        Mask mask(proto_w, proto_h);
        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                float val = 0.0f;
                for (int c = 0; c < kNumMasks; c++) {
                    val += proto[(size_t)c * proto_h * proto_w + (size_t)h * proto_w + w] * coeff[c];
                }
                mask.data[(size_t)h * proto_w + w] = sigmoid(val);
            }
        }

        // 裁剪到框区域（proto mask 空间）
        // 注：沿用历史实现的坐标换算（对已还原到原图的 box 再减一次 padding），
        // 该换算本身存疑，见 AGENTS.md Gotchas，本次迁移保持行为不变。
        const float proto_scale = (float)proto_w / std::max(1, input_width);
        int x1 = std::max(0, (int)((box.x1 - lb.pad_left) * proto_scale));
        int y1 = std::max(0, (int)((box.y1 - lb.pad_top)  * proto_scale));
        int x2 = std::min(proto_w - 1, (int)((box.x2 - lb.pad_left) * proto_scale));
        int y2 = std::min(proto_h - 1, (int)((box.y2 - lb.pad_top)  * proto_scale));

        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                if (w < x1 || w > x2 || h < y1 || h > y2) {
                    mask.data[(size_t)h * proto_w + w] = 0.0f;
                }
            }
        }

        // 缩放到原图尺寸并二值化
        cv::Mat mask_cv(proto_h, proto_w, CV_32FC1, mask.data.data());
        cv::Mat resized, binary;
        cv::resize(mask_cv, resized, cv::Size(orig_w, orig_h));
        cv::threshold(resized, binary, 0.5f, 1.0f, cv::THRESH_BINARY);

        Mask final_mask(orig_w, orig_h);
        for (int h = 0; h < orig_h; h++) {
            for (int w = 0; w < orig_w; w++) {
                final_mask.data[(size_t)h * orig_w + w] = binary.at<float>(h, w);
            }
        }
        masks.push_back(std::move(final_mask));
    }
    return masks;
}

InferResult PostProcessSegment::forward(const TensorSet& outputs,
                                        const LetterboxInfo& lb) const {
    auto cands = decoder_->decode_segment(outputs, ctx_);
    if (cands.empty()) return SegmentResult{};

    std::vector<Box> boxes;
    boxes.reserve(cands.size());
    for (const auto& c : cands) boxes.push_back(c.box);

    auto keep = select(boxes, lb);

    SegmentResult result;
    result.boxes.reserve(keep.size());
    std::vector<std::array<float, kNumMasks>> kept_coeffs;
    kept_coeffs.reserve(keep.size());

    for (int i : keep) {
        Box b = cands[i].box;
        scale_box(b, lb);
        result.boxes.push_back(b);
        kept_coeffs.push_back(cands[i].extra);
    }

    const int proto_idx = find_proto_index(outputs);
    if (proto_idx >= 0 && !result.boxes.empty()) {
        result.masks = compose_masks(outputs, proto_idx, kept_coeffs, lb, result.boxes,
                                     ctx_.input_width);
    }
    return result;
}

// ============================================================
// 工厂：按任务装配（自动选好对应 Decoder）
// ============================================================
std::shared_ptr<PostProcess> create_postprocess(TaskType task, ModelType model_type,
                                                const PostProcessParams& params) {
    auto decoder = create_decoder(model_type);
    if (!decoder) {
        std::cerr << "[create_postprocess] No decoder for model type: "
                  << model_type_name(model_type) << std::endl;
        return nullptr;
    }

    switch (task) {
        case TaskType::Detect:  return std::make_shared<PostProcessDet>(params, decoder);
        case TaskType::Segment: return std::make_shared<PostProcessSegment>(params, decoder);
        case TaskType::Pose:    return std::make_shared<PostProcessPose>(params, decoder);
        case TaskType::OBB:     return std::make_shared<PostProcessOBB>(params, decoder);
        default:
            std::cerr << "[create_postprocess] Unknown task type: "
                      << task_type_name(task) << std::endl;
            return nullptr;
    }
}

} // namespace yolo_onnx