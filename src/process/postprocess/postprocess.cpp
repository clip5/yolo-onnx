#include "process/postprocess/postprocess.hpp"
#include <opencv2/imgproc.hpp>

namespace yolo_onnx {

// ============================================================
// IoU / NMS（普通框）— 迁自 yolo_onnx_types.hpp 中的 inline 实现
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
// 检测后处理：还原坐标 + NMS
// ============================================================
DetectResult postprocess_detect(BoxArray boxes, const LetterboxInfo& info,
                                float nms_thresh) {
    for (auto& box : boxes) {
        scale_box(box, info);
    }
    DetectResult result;
    auto keep = nms(boxes, nms_thresh);
    result.boxes.reserve(keep.size());
    for (int idx : keep) {
        result.boxes.push_back(boxes[idx]);
    }
    return result;
}

// OBB 后处理：还原坐标 + 过滤 + 旋转 NMS（obb_nms 由 model_v8_obb 提供时复用；
// 这里提供基于旋转 IoU 的默认实现）
OBBResult postprocess_obb(OBBBoxArray boxes, const LetterboxInfo& info,
                          float nms_thresh) {
    for (auto& box : boxes) {
        scale_obb_box(box, info);
    }
    OBBResult result;
    auto keep = obb_nms(boxes, nms_thresh);
    result.obb_boxes.reserve(keep.size());
    for (int idx : keep) {
        result.obb_boxes.push_back(boxes[idx]);
    }
    return result;
}

} // namespace yolo_onnx
