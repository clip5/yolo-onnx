#include "process/postprocess/postprocess_core.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <map>

namespace yolo_onnx {

// ============================================================
// 1) 基元：IoU / NMS
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

namespace {

/// 通用 NMS 核心：@p order 为待处理下标（调用方保证已按分数降序）
template <typename Overlap>
std::vector<int> nms_impl(const std::vector<int>& order, float iou_threshold,
                          Overlap&& overlap) {
    std::vector<int> indices;
    std::vector<bool> removed(order.size(), false);
    for (int i = 0; i < (int)order.size(); i++) {
        if (removed[order[i]]) continue;
        indices.push_back(order[i]);
        for (int j = i + 1; j < (int)order.size(); j++) {
            if (removed[order[j]]) continue;
            if (overlap(order[i], order[j]) > iou_threshold) {
                removed[order[j]] = true;
            }
        }
    }
    return indices;
}

/// 生成按分数降序的下标序列
template <typename Arr>
std::vector<int> sorted_by_score(const Arr& boxes) {
    std::vector<int> order(boxes.size());
    for (int i = 0; i < (int)boxes.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return boxes[i].score > boxes[j].score; });
    return order;
}

} // namespace

std::vector<int> nms(const BoxArray& boxes, float iou_threshold, NmsMode mode) {
    if (mode == NmsMode::Agnostic) {
        return nms_impl(sorted_by_score(boxes), iou_threshold,
                        [&](int i, int j) { return iou(boxes[i], boxes[j]); });
    }

    // ClassAware：按类别分组，组内独立 NMS，跨组互不抑制。
    // 早期实现把所有类别混在一起做 NMS，会让「重叠的不同类目标」被误抑制
    // （例如紧邻的人与自行车），这是 YOLOv5 起改为按类别分组的原因。
    std::map<int, std::vector<int>> by_class;
    for (int i = 0; i < (int)boxes.size(); i++) {
        by_class[boxes[i].label].push_back(i);
    }

    std::vector<int> kept;
    for (const auto& kv : by_class) {
        BoxArray subset;
        subset.reserve(kv.second.size());
        for (int i : kv.second) subset.push_back(boxes[i]);

        // 组内 NMS 返回 subset 的下标，映射回原始下标
        const auto order = sorted_by_score(subset);
        for (int local : nms_impl(order, iou_threshold,
                                  [&](int i, int j) { return iou(subset[i], subset[j]); })) {
            kept.push_back(kv.second[local]);
        }
    }

    // 各组结果合并后统一按分数降序（组内已降序，跨组需重排）
    std::sort(kept.begin(), kept.end(),
              [&](int i, int j) { return boxes[i].score > boxes[j].score; });
    return kept;
}

std::vector<int> nms_class_aware(const BoxArray& boxes, float iou_threshold) {
    return nms(boxes, iou_threshold, NmsMode::ClassAware);
}

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
    return nms_impl(sorted_by_score(boxes), iou_threshold,
                    [&](int i, int j) { return obb_iou(boxes[i], boxes[j]); });
}

// ============================================================
// 2) 步骤：坐标还原 / 过滤 / 截断
// ============================================================

void restore_boxes(BoxArray& boxes, const LetterboxInfo& lb) {
    for (auto& box : boxes) scale_box(box, lb);
}

void restore_keypoints(std::vector<Keypoint>& kps, const LetterboxInfo& lb) {
    for (auto& kp : kps) scale_keypoint(kp, lb);
}

void restore_obb(OBBBoxArray& boxes, const LetterboxInfo& lb) {
    for (auto& box : boxes) scale_obb_box(box, lb);
}

BoxArray filter_by_score(const BoxArray& boxes, float score_thresh) {
    if (score_thresh <= 0.0f) return boxes;
    BoxArray out;
    out.reserve(boxes.size());
    for (const auto& b : boxes) {
        if (b.score >= score_thresh) out.push_back(b);
    }
    return out;
}

OBBBoxArray filter_by_score(const OBBBoxArray& boxes, float score_thresh) {
    if (score_thresh <= 0.0f) return boxes;
    OBBBoxArray out;
    out.reserve(boxes.size());
    for (const auto& b : boxes) {
        if (b.score >= score_thresh) out.push_back(b);
    }
    return out;
}

BoxArray top_k(BoxArray boxes, int max_detections) {
    if (max_detections <= 0 || (int)boxes.size() <= max_detections) return boxes;
    std::partial_sort(boxes.begin(), boxes.begin() + max_detections, boxes.end(),
                      [](const Box& a, const Box& b) { return a.score > b.score; });
    boxes.resize(max_detections);
    return boxes;
}

OBBBoxArray top_k(OBBBoxArray boxes, int max_detections) {
    if (max_detections <= 0 || (int)boxes.size() <= max_detections) return boxes;
    std::partial_sort(boxes.begin(), boxes.begin() + max_detections, boxes.end(),
                      [](const OBBBox& a, const OBBBox& b) { return a.score > b.score; });
    boxes.resize(max_detections);
    return boxes;
}

// ============================================================
// 3) 组合：一站式后处理
// ============================================================

std::vector<int> select_indices(const std::vector<Box>& candidates,
                                const LetterboxInfo& lb,
                                const PostProcessParams& params) {
    // 1. 置信度过滤（保留原始下标）
    std::vector<int> idx;
    idx.reserve(candidates.size());
    for (int i = 0; i < (int)candidates.size(); i++) {
        if (params.score_thresh <= 0.0f || candidates[i].score >= params.score_thresh) {
            idx.push_back(i);
        }
    }
    if (idx.empty()) return {};

    // 2. 还原到原图坐标系后再做 NMS（模型输入尺度与原图尺度不同，
    //    直接用未还原的框算 IoU 会得到错误的抑制关系）
    BoxArray restored;
    restored.reserve(idx.size());
    for (int i : idx) {
        Box b = candidates[i];
        scale_box(b, lb);
        restored.push_back(b);
    }

    // 3. NMS（返回 restored 的下标）
    auto kept = nms(restored, params.nms_thresh, params.nms_mode);

    // 4. 映射回候选下标
    std::vector<int> keep;
    keep.reserve(kept.size());
    for (int k : kept) keep.push_back(idx[k]);

    // 5. 数量限制（keep 已是分数降序，可直接截断）
    if (params.max_detections > 0 && (int)keep.size() > params.max_detections) {
        keep.resize(params.max_detections);
    }
    return keep;
}

BoxArray detect_pipeline(BoxArray candidates, const LetterboxInfo& lb,
                         const PostProcessParams& params) {
    if (candidates.empty()) return {};

    auto keep = select_indices(candidates, lb, params);

    BoxArray out;
    out.reserve(keep.size());
    for (int i : keep) {
        Box b = candidates[i];
        scale_box(b, lb);
        out.push_back(b);
    }
    return out;
}

OBBBoxArray obb_pipeline(OBBBoxArray candidates, const LetterboxInfo& lb,
                         const PostProcessParams& params) {
    if (candidates.empty()) return {};

    auto boxes = filter_by_score(candidates, params.score_thresh);
    restore_obb(boxes, lb);

    auto keep = obb_nms(boxes, params.nms_thresh);
    OBBBoxArray out;
    out.reserve(keep.size());
    for (int k : keep) out.push_back(boxes[k]);

    return top_k(std::move(out), params.max_detections);
}

} // namespace yolo_onnx