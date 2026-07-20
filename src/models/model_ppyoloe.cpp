#include "yolo_onnx/models/model_ppyoloe.hpp"
#include "yolo_onnx/backend.hpp"
#include <iostream>

namespace yolo_onnx {

InferResult ModelPPYOLOE::infer(const cv::Mat& image) {
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
        std::cerr << "[ModelPPYOLOE] Inference failed" << std::endl;
        return DetectResult{};
    }

    // 4. Decode output
    BoxArray boxes = decode_output(output_data, output_shapes);

    // 5. Scale boxes back to original image
    for (auto& box : boxes) {
        scale_box(box, pre.letterbox);
    }

    // 6. NMS
    auto keep = nms(boxes, config_.nms_thresh);
    BoxArray result;
    for (int idx : keep) {
        result.push_back(boxes[idx]);
    }

    return DetectResult{result};
}

BoxArray ModelPPYOLOE::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    BoxArray boxes;
    int num_outputs = (int)output_data.size();
    if (num_outputs == 0) return boxes;

    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;

    // PPYOLOE common output formats:
    //   Case 1: 6 outputs — 3 cls + 3 reg (one per FPN level)
    //   Case 2: 3 outputs — each [1, C+4, H, W] (combined cls+reg per level)
    //   Case 3: 1 output — [1, ?, 8400] (concatenated)

    if (num_outputs == 6) {
        // 6 outputs: [cls0, cls1, cls2, reg0, reg1, reg2]
        // Each pair corresponds to stride 8, 16, 32
        for (int level = 0; level < 3; level++) {
            const auto& cls_data = output_data[level];
            const auto& reg_data = output_data[level + 3];
            const auto& cls_shape = output_shapes[level];
            const auto& reg_shape = output_shapes[level + 3];

            if (cls_shape.size() != 4 || reg_shape.size() != 4) continue;
            int cls_h = (int)cls_shape[2];
            int cls_w = (int)cls_shape[3];
            int reg_h = (int)reg_shape[2];
            int reg_w = (int)reg_shape[3];
            int stride = strides_[level];

            // cls_shape: [1, C, H, W]
            // reg_shape: [1, 4, H, W]
            // PPYOLOE uses sigmoid for cls, no separate obj_conf

            for (int h = 0; h < cls_h; h++) {
                for (int w = 0; w < cls_w; w++) {
                    // Find max class score
                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float cls = sigmoid(cls_data[c * cls_h * cls_w + h * cls_w + w]);
                        if (cls > max_cls) {
                            max_cls = cls;
                            max_cls_id = c;
                        }
                    }

                    if (max_cls < score_thresh) continue;

                    // Decode box (anchor-free, similar to YOLOX)
                    // PPYOLOE: reg = [dx, dy, dw, dh]
                    // dx, dy: sigmoid + grid offset
                    // dw, dh: exp * stride
                    float dx = reg_data[0 * reg_h * reg_w + h * reg_w + w];
                    float dy = reg_data[1 * reg_h * reg_w + h * reg_w + w];
                    float dw = reg_data[2 * reg_h * reg_w + h * reg_w + w];
                    float dh = reg_data[3 * reg_h * reg_w + h * reg_w + w];

                    float cx = (sigmoid(dx) + w) * stride;
                    float cy = (sigmoid(dy) + h) * stride;
                    float bw = std::exp(dw) * stride;
                    float bh = std::exp(dh) * stride;

                    float x1 = cx - bw / 2.0f;
                    float y1 = cy - bh / 2.0f;
                    float x2 = cx + bw / 2.0f;
                    float y2 = cy + bh / 2.0f;

                    boxes.emplace_back(x1, y1, x2, y2, max_cls, max_cls_id);
                }
            }
        }
    }
    else if (num_outputs == 3) {
        // 3 outputs: each [1, 4+C, H, W]
        for (int level = 0; level < 3; level++) {
            const auto& data = output_data[level];
            const auto& shape = output_shapes[level];

            if (shape.size() != 4) continue;
            int channels = (int)shape[1];
            int height   = (int)shape[2];
            int width    = (int)shape[3];
            int stride   = strides_[level];

            if (channels < 4 + num_classes) continue;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float cls = sigmoid(data[(4 + c) + h * width + w]);
                        if (cls > max_cls) {
                            max_cls = cls;
                            max_cls_id = c;
                        }
                    }

                    if (max_cls < score_thresh) continue;

                    float dx = data[0 + h * width + w];
                    float dy = data[1 + h * width + w];
                    float dw = data[2 + h * width + w];
                    float dh = data[3 + h * width + w];

                    float cx = (sigmoid(dx) + w) * stride;
                    float cy = (sigmoid(dy) + h) * stride;
                    float bw = std::exp(dw) * stride;
                    float bh = std::exp(dh) * stride;

                    float x1 = cx - bw / 2.0f;
                    float y1 = cy - bh / 2.0f;
                    float x2 = cx + bw / 2.0f;
                    float y2 = cy + bh / 2.0f;

                    boxes.emplace_back(x1, y1, x2, y2, max_cls, max_cls_id);
                }
            }
        }
    }
    else if (num_outputs == 1) {
        // 1 concatenated output: [1, ?, 8400]
        const auto& data = output_data[0];
        const auto& shape = output_shapes[0];

        if (shape.size() != 3 && shape.size() != 2) {
            std::cerr << "[ModelPPYOLOE] Unexpected single output shape" << std::endl;
            return boxes;
        }

        int num_boxes = (int)shape.back();
        int channels = (shape.size() == 3) ? (int)shape[1] : num_classes + 4;

        // Assume concatenated format: [1, 4+C, 8400] or [1, 8400, 4+C]
        // Try 8400 grid decomposition
        int grid_counts[3] = {6400, 1600, 400};
        int grid_sizes[3] = {80, 40, 20};
        int grid_offset = 0;

        // Heuristic: determine if shape is [1, C, N] or [1, N, C]
        bool is_chw = (shape.size() == 3); // [1, C, N]

        for (int level = 0; level < 3; level++) {
            int stride = strides_[level];
            int grid_h = grid_sizes[level];
            int grid_w = grid_sizes[level];

            for (int g = 0; g < grid_counts[level]; g++) {
                int idx = grid_offset + g;

                if (is_chw) {
                    // [1, 4+C, N] format
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

                    float dx = data[0 * num_boxes + idx];
                    float dy = data[1 * num_boxes + idx];
                    float dw = data[2 * num_boxes + idx];
                    float dh = data[3 * num_boxes + idx];

                    int gi = idx % grid_w;
                    int gj = idx / grid_w;

                    float cx = (sigmoid(dx) + gi) * stride;
                    float cy = (sigmoid(dy) + gj) * stride;
                    float bw = std::exp(dw) * stride;
                    float bh = std::exp(dh) * stride;

                    float x1 = cx - bw / 2.0f;
                    float y1 = cy - bh / 2.0f;
                    float x2 = cx + bw / 2.0f;
                    float y2 = cy + bh / 2.0f;

                    boxes.emplace_back(x1, y1, x2, y2, max_cls, max_cls_id);
                }
            }
            grid_offset += grid_counts[level];
        }
    }

    return boxes;
}

} // namespace yolo_onnx