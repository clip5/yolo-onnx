#include "yolo_onnx/models/model_v8.hpp"
#include <iostream>

namespace yolo_onnx {

BoxArray ModelV8::infer(const cv::Mat& image) {
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
        std::cerr << "[ModelV8] Inference failed" << std::endl;
        return {};
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

    return result;
}

BoxArray ModelV8::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    BoxArray boxes;
    if (output_data.empty()) return boxes;

    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;

    // YOLOv8/v11/v26: single output [1, 4+num_classes(+reg_max*16), num_boxes]
    // or sometimes 3 outputs (one per stride level)
    // Handle both cases

    const auto& data = output_data[0];
    const auto& shape = output_shapes[0];

    // Case 1: Single concatenated output [1, C, N]
    if (shape.size() == 3) {
        int channels = (int)shape[1];
        int num_boxes = (int)shape[2];

        // Check if this is DFL format (channels = 4*16 + C) or direct (channels = 4 + C)
        bool has_dfl = (channels > 4 + num_classes + 32);  // heuristic: if C > 84+32 for COCO
        int reg_max = 0;
        int cls_start = 4;
        if (has_dfl) {
            reg_max = (channels - num_classes) / 4;
            cls_start = 4 * reg_max;
        }

        // Build grid info: 80*80 + 40*40 + 20*20 = 8400
        int grid_sizes[3] = {80, 40, 20};
        int grid_counts[3] = {6400, 1600, 400};
        int grid_offset = 0;

        for (int level = 0; level < 3; level++) {
            int stride = strides_[level];
            int grid_h = grid_sizes[level];
            int grid_w = grid_sizes[level];

            for (int g = 0; g < grid_counts[level]; g++) {
                int idx = grid_offset + g;
                int gi = idx % grid_w;
                int gj = idx / grid_w;

                // Class scores (need sigmoid)
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
                float cx, cy, bw, bh;

                if (has_dfl) {
                    // DFL decode: for each of 4 sides, softmax over reg_max values
                    float dfl_val[4] = {0};
                    for (int s = 0; s < 4; s++) {
                        float sum_exp = 0.0f;
                        float probs[16] = {0};
                        for (int i = 0; i < reg_max; i++) {
                            probs[i] = std::exp(data[(s * reg_max + i) * num_boxes + idx]);
                            sum_exp += probs[i];
                        }
                        if (sum_exp > 0) {
                            for (int i = 0; i < reg_max; i++) {
                                dfl_val[s] += (probs[i] / sum_exp) * i;
                            }
                        }
                    }
                    // dfl_val: [left, top, right, bottom] offsets from grid center
                    cx = (gi + 0.5f) * stride;
                    cy = (gj + 0.5f) * stride;
                    bw = (dfl_val[0] + dfl_val[2]) * stride;
                    bh = (dfl_val[1] + dfl_val[3]) * stride;
                } else {
                    // Direct decode: cx, cy, w, h in grid space
                    cx = data[0 * num_boxes + idx];
                    cy = data[1 * num_boxes + idx];
                    bw = data[2 * num_boxes + idx];
                    bh = data[3 * num_boxes + idx];

                    // The values might be raw (need sigmoid + grid offset) or already decoded
                    // Check if they are in reasonable range (0-1 vs 0-640)
                    if (cx >= 0.0f && cx <= 1.0f && cy >= 0.0f && cy <= 1.0f) {
                        // Raw values: need to decode
                        cx = (sigmoid(cx) + gi) * stride;
                        cy = (sigmoid(cy) + gj) * stride;
                        bw = bw * stride;  // might need exp(bw) * stride
                        bh = bh * stride;
                    } else {
                        // Already decoded, just scale by stride
                        cx = cx * stride;
                        cy = cy * stride;
                        bw = bw * stride;
                        bh = bh * stride;
                    }
                }

                float x1 = cx - bw / 2.0f;
                float y1 = cy - bh / 2.0f;
                float x2 = cx + bw / 2.0f;
                float y2 = cy + bh / 2.0f;

                boxes.emplace_back(x1, y1, x2, y2, max_cls, max_cls_id);
            }
            grid_offset += grid_counts[level];
        }
    }
    // Case 2: 3 separate outputs (one per stride level)
    else if (output_data.size() >= 3 && shape.size() == 4) {
        for (int level = 0; level < 3; level++) {
            const auto& lvl_data = output_data[level];
            const auto& lvl_shape = output_shapes[level];

            if (lvl_shape.size() != 4) continue;
            int channels = (int)lvl_shape[1];
            int height   = (int)lvl_shape[2];
            int width    = (int)lvl_shape[3];
            int stride   = strides_[level];

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float cls = sigmoid(lvl_data[(4 + c) + h * width + w]);
                        if (cls > max_cls) {
                            max_cls = cls;
                            max_cls_id = c;
                        }
                    }

                    if (max_cls < score_thresh) continue;

                    float cx = lvl_data[0 + h * width + w];
                    float cy = lvl_data[1 + h * width + w];
                    float bw = lvl_data[2 + h * width + w];
                    float bh = lvl_data[3 + h * width + w];

                    // Raw values: decode
                    float bx = (sigmoid(cx) + w) * stride;
                    float by = (sigmoid(cy) + h) * stride;
                    float box_w = bw * stride;
                    float box_h = bh * stride;

                    float x1 = bx - box_w / 2.0f;
                    float y1 = by - box_h / 2.0f;
                    float x2 = bx + box_w / 2.0f;
                    float y2 = by + box_h / 2.0f;

                    boxes.emplace_back(x1, y1, x2, y2, max_cls, max_cls_id);
                }
            }
        }
    }

    return boxes;
}

} // namespace yolo_onnx