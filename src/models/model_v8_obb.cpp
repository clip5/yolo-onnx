#include "yolo_onnx/models/model_v8_obb.hpp"
#include "yolo_onnx/backend.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// OBB utility functions (local to this translation unit)
// ============================================================
static float obb_iou(const OBBBox& a, const OBBBox& b) {
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

static std::vector<int> obb_nms(const OBBBoxArray& boxes, float iou_threshold) {
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

InferResult ModelV8OBB::infer(const cv::Mat& image) {
    return infer_obb(image);
}

OBBResult ModelV8OBB::infer_obb(const cv::Mat& image) {
    OBBResult result;

    // 1. Preprocess
    auto pre = preprocess(image, config_.input_width, config_.input_height);

    // 2. Prepare backend input
    auto input_names = backend_->get_input_names();
    std::vector<int64_t> input_shape = {1, 3, config_.input_height, config_.input_width};
    std::vector<std::vector<int64_t>> input_shapes = {input_shape};

    // 3. Run inference
    auto output_names = backend_->get_output_names();
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::vector<float>> output_data;

    bool ok = backend_->forward(
        input_names, input_shapes, pre.blob,
        output_names, output_shapes, output_data
    );

    if (!ok) {
        std::cerr << "[ModelV8OBB] Inference failed" << std::endl;
        return result;
    }

    // 4. Decode OBB candidates
    auto obb_boxes = decode_obb_candidates(output_data, output_shapes);

    // 5. Scale OBB boxes back to original image
    for (auto& obb : obb_boxes) {
        scale_obb_box(obb, pre.letterbox);
    }

    // 6. OBB NMS (uses rotated IoU)
    auto keep = obb_nms(obb_boxes, config_.nms_thresh);
    for (int idx : keep) {
        result.obb_boxes.push_back(obb_boxes[idx]);
    }

    return result;
}

BoxArray ModelV8OBB::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    auto obb_boxes = decode_obb_candidates(output_data, output_shapes);
    BoxArray boxes;
    for (const auto& obb : obb_boxes) {
        boxes.push_back(obb.aabb());
    }
    return boxes;
}

OBBBoxArray ModelV8OBB::decode_obb_candidates(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    OBBBoxArray candidates;
    if (output_data.empty()) return candidates;

    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;

    const auto& data = output_data[0];
    const auto& shape = output_shapes[0];

    if (shape.size() < 3) return candidates;

    int channels, num_boxes, height, width;
    bool is_chw_layout = false;

    if (shape.size() == 3) {
        channels  = (int)shape[1];
        num_boxes = (int)shape[2];
        height = width = 0;
    } else if (shape.size() == 4) {
        channels = (int)shape[1];
        height   = (int)shape[2];
        width    = (int)shape[3];
        num_boxes = 0;
        is_chw_layout = true;
    } else {
        return candidates;
    }

    // YOLOv8-obb: [cx, cy, w, h, cls0, cls1, ..., angle]
    // or [cx, cy, w, h, angle, cls0, cls1, ...]
    // The angle is 1 extra channel. Total channels = 4 + num_classes + 1
    // Determine if angle is before or after classes
    // Heuristic: if channels == 4 + 1 + num_classes, angle is at position 4 (after box, before cls)
    // or at position 4 + num_classes (after box + cls)
    int angle_pos = 4;  // default: angle at position 4 (before classes)
    if (channels == 4 + num_classes + 1) {
        // Try both positions
        // Usually angle is at position 4 (before classes) in YOLOv8-obb
        angle_pos = 4;
    }
    int cls_start = 4;
    // If angle is before classes, classes start at 5
    if (channels >= 4 + 1 + num_classes) {
        // Check: if there's a value at position 4 that looks like an angle
        // (not a probability), then angle is at 4 and classes start at 5
        // We'll use angle_pos = 4, cls_start = 5
        cls_start = 5;
    }

    if (is_chw_layout) {
        // 4D output: [1, C, H, W]
        // Each grid cell corresponds to one stride level
        int stride = config_.input_width / width;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    float cls = sigmoid(data[(cls_start + c) * height * width + h * width + w]);
                    if (cls > max_cls) {
                        max_cls = cls;
                        max_cls_id = c;
                    }
                }
                if (max_cls < score_thresh) continue;

                // Decode box
                float cx = data[0 * height * width + h * width + w];
                float cy = data[1 * height * width + h * width + w];
                float bw = data[2 * height * width + h * width + w];
                float bh = data[3 * height * width + h * width + w];
                float angle = sigmoid(data[angle_pos * height * width + h * width + w]) * CV_PI;

                float bx = (sigmoid(cx) + w) * stride;
                float by = (sigmoid(cy) + h) * stride;
                float box_w = bw * stride;
                float box_h = bh * stride;

                candidates.emplace_back(bx, by, box_w, box_h, angle, max_cls, max_cls_id);
            }
        }
    } else {
        // 3D output [1, C, N] — concatenated grid
        int grid_sizes[3] = {80, 40, 20};
        int grid_counts[3] = {6400, 1600, 400};
        int grid_offset = 0;

        for (int level = 0; level < 3; level++) {
            int stride = strides_[level];
            int grid_w = grid_sizes[level];

            for (int g = 0; g < grid_counts[level]; g++) {
                int idx = grid_offset + g;
                int gi = idx % grid_w;
                int gj = idx / grid_w;

                // Class scores
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    float cls = sigmoid(data[(cls_start + c) * num_boxes + idx]);
                    if (cls > max_cls) {
                        max_cls = cls;
                        max_cls_id = c;
                    }
                }
                if (max_cls < score_thresh) continue;

                // Decode box
                float cx = data[0 * num_boxes + idx];
                float cy = data[1 * num_boxes + idx];
                float bw = data[2 * num_boxes + idx];
                float bh = data[3 * num_boxes + idx];
                float angle = sigmoid(data[angle_pos * num_boxes + idx]) * CV_PI;

                float bx, by, box_w, box_h;
                if (cx >= 0.0f && cx <= 1.0f && cy >= 0.0f && cy <= 1.0f) {
                    bx = (sigmoid(cx) + gi) * stride;
                    by = (sigmoid(cy) + gj) * stride;
                    box_w = bw * stride;
                    box_h = bh * stride;
                } else {
                    bx = cx * stride;
                    by = cy * stride;
                    box_w = bw * stride;
                    box_h = bh * stride;
                }

                candidates.emplace_back(bx, by, box_w, box_h, angle, max_cls, max_cls_id);
            }
            grid_offset += grid_counts[level];
        }
    }

    return candidates;
}

} // namespace yolo_onnx