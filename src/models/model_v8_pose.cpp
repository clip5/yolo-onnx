#include "yolo_onnx/models/model_v8_pose.hpp"
#include "yolo_onnx/backend.hpp"
#include <iostream>

namespace yolo_onnx {

InferResult ModelV8Pose::infer(const cv::Mat& image) {
    return infer_pose(image);
}

PoseResult ModelV8Pose::infer_pose(const cv::Mat& image) {
    PoseResult result;

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
        std::cerr << "[ModelV8Pose] Inference failed" << std::endl;
        return result;
    }

    // 4. Decode candidates (boxes + keypoints)
    auto candidates = decode_pose_candidates(output_data, output_shapes);

    // 5. Scale boxes and keypoints back to original image
    for (auto& cand : candidates) {
        scale_box(cand.box, pre.letterbox);
        for (auto& kp : cand.keypoints) {
            scale_keypoint(kp, pre.letterbox);
        }
    }

    // 6. NMS on boxes
    BoxArray boxes;
    for (const auto& cand : candidates) {
        boxes.push_back(cand.box);
    }
    auto keep = nms(boxes, config_.nms_thresh);

    // 7. Collect results
    for (int idx : keep) {
        result.boxes.push_back(candidates[idx].box);
        result.keypoints.push_back(candidates[idx].keypoints);
    }

    return result;
}

BoxArray ModelV8Pose::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    auto candidates = decode_pose_candidates(output_data, output_shapes);
    BoxArray boxes;
    for (const auto& cand : candidates) {
        boxes.push_back(cand.box);
    }
    return boxes;
}

std::vector<ModelV8Pose::DecodedPoseBox> ModelV8Pose::decode_pose_candidates(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    std::vector<DecodedPoseBox> candidates;
    if (output_data.empty()) return candidates;

    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;
    int num_kpts = config_.num_keypoints;

    const auto& data = output_data[0];
    const auto& shape = output_shapes[0];

    if (shape.size() < 3) return candidates;

    int channels, num_boxes;
    bool is_chw_layout = false;

    if (shape.size() == 3) {
        channels  = (int)shape[1];
        num_boxes = (int)shape[2];
    } else if (shape.size() == 4) {
        channels = (int)shape[1];
        num_boxes = 0;
        is_chw_layout = true;
    } else {
        return candidates;
    }

    int kpt_start = 4 + num_classes;  // where keypoint data begins
    int kpt_dim = 3;  // x, y, visibility per keypoint

    if (is_chw_layout) {
        // 4D output: [1, C, H, W] — treat as grids per stride
        int height = (int)shape[2];
        int width  = (int)shape[3];
        int stride = config_.input_width / width;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    float cls = sigmoid(data[(4 + c) * height * width + h * width + w]);
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

                float bx = (sigmoid(cx) + w) * stride;
                float by = (sigmoid(cy) + h) * stride;
                float box_w = bw * stride;
                float box_h = bh * stride;

                float x1 = bx - box_w / 2.0f;
                float y1 = by - box_h / 2.0f;
                float x2 = bx + box_w / 2.0f;
                float y2 = by + box_h / 2.0f;

                DecodedPoseBox cand;
                cand.box = Box(x1, y1, x2, y2, max_cls, max_cls_id);

                // Decode keypoints
                for (int k = 0; k < num_kpts; k++) {
                    Keypoint kp;
                    kp.x = data[(kpt_start + k * kpt_dim + 0) * height * width + h * width + w];
                    kp.y = data[(kpt_start + k * kpt_dim + 1) * height * width + h * width + w];
                    kp.visibility = data[(kpt_start + k * kpt_dim + 2) * height * width + h * width + w];

                    // Decode x, y from grid space
                    kp.x = (sigmoid(kp.x) + w) * stride;
                    kp.y = (sigmoid(kp.y) + h) * stride;

                    cand.keypoints.push_back(kp);
                }
                candidates.push_back(cand);
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
                    float cls = sigmoid(data[(4 + c) * num_boxes + idx]);
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

                float x1 = bx - box_w / 2.0f;
                float y1 = by - box_h / 2.0f;
                float x2 = bx + box_w / 2.0f;
                float y2 = by + box_h / 2.0f;

                DecodedPoseBox cand;
                cand.box = Box(x1, y1, x2, y2, max_cls, max_cls_id);

                // Decode keypoints
                for (int k = 0; k < num_kpts; k++) {
                    Keypoint kp;
                    kp.x = data[(kpt_start + k * kpt_dim + 0) * num_boxes + idx];
                    kp.y = data[(kpt_start + k * kpt_dim + 1) * num_boxes + idx];
                    kp.visibility = data[(kpt_start + k * kpt_dim + 2) * num_boxes + idx];

                    // Decode x, y from grid space
                    if (kp.x >= 0.0f && kp.x <= 1.0f && kp.y >= 0.0f && kp.y <= 1.0f) {
                        kp.x = (sigmoid(kp.x) + gi) * stride;
                        kp.y = (sigmoid(kp.y) + gj) * stride;
                    } else {
                        kp.x = kp.x * stride;
                        kp.y = kp.y * stride;
                    }

                    cand.keypoints.push_back(kp);
                }
                candidates.push_back(cand);
            }
            grid_offset += grid_counts[level];
        }
    }

    return candidates;
}

} // namespace yolo_onnx