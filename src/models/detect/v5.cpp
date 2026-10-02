#include "models/detect/v5.hpp"
#include "core/backend.hpp"
#include <iostream>

namespace yolo_onnx {

InferResult ModelV5::infer(const cv::Mat& image) {
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
        std::cerr << "[ModelV5] Inference failed" << std::endl;
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

BoxArray ModelV5::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    BoxArray boxes;
    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;
    // Case 1: Single concatenated output [1, 25200, 5+C] from ultralytics
    // export — anchors flattened and box values already decoded to pixels,
    // obj/class scores already sigmoid'd.
    if (output_data.size() == 1) {
        const auto& data = output_data[0];
        const auto& shape = output_shapes[0];
        if (shape.size() != 3 || (int)shape[2] != 5 + num_classes) {
            std::cerr << "[ModelV5] Unexpected single output shape" << std::endl;
            return boxes;
        }
        int num_boxes = (int)shape[1];
        int bbox_ch = 5 + num_classes;

        for (int i = 0; i < num_boxes; i++) {
            const float* row = &data[(size_t)i * bbox_ch];
            float obj = row[4];
            float max_cls = 0.0f;
            int max_cls_id = -1;
            for (int c = 0; c < num_classes; c++) {
                if (row[5 + c] > max_cls) {
                    max_cls = row[5 + c];
                    max_cls_id = c;
                }
            }
            float score = obj * max_cls;
            if (max_cls_id < 0 || score < score_thresh) continue;

            float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
            boxes.emplace_back(cx - bw / 2.0f, cy - bh / 2.0f,
                               cx + bw / 2.0f, cy + bh / 2.0f,
                               score, max_cls_id);
        }
        return boxes;
    }

    // Case 2: raw per-level outputs [1, 3*(5+C), H, W]
    if (output_data.size() != 3) {
        std::cerr << "[ModelV5] Expected 1 or 3 outputs, got " << output_data.size() << std::endl;
        return boxes;
    }

    // Process each of the 3 output scales (strides 8, 16, 32)
    for (int level = 0; level < 3; level++) {
        const auto& data = output_data[level];
        const auto& shape = output_shapes[level];

        // Expected shape: [1, 3*(5+C), H, W]
        if (shape.size() != 4) continue;
        int channels = (int)shape[1];
        int height   = (int)shape[2];
        int width    = (int)shape[3];
        int stride   = strides_[level];

        int num_anchors = 3;
        int bbox_ch = 5 + num_classes;  // 5 = tx, ty, tw, th, obj

        if (channels != num_anchors * bbox_ch) {
            // Might be a different export format; try to adapt
            continue;
        }

        int hw = height * width;

        for (int a = 0; a < num_anchors; a++) {
            int anchor_offset = a * bbox_ch;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // Read obj confidence
                    float obj = sigmoid(data[anchor_offset + 4 + h * width + w]);

                    // Find max class score
                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float cls = sigmoid(data[anchor_offset + 5 + c + h * width + w]);
                        if (cls > max_cls) {
                            max_cls = cls;
                            max_cls_id = c;
                        }
                    }

                    float score = obj * max_cls;
                    if (score < score_thresh) continue;

                    // Decode box
                    int idx_off = anchor_offset + h * width + w;
                    float tx = data[idx_off];
                    float ty = data[anchor_offset + 1 + h * width + w];
                    float tw = data[anchor_offset + 2 + h * width + w];
                    float th = data[anchor_offset + 3 + h * width + w];

                    float bx = (sigmoid(tx) + w) * stride;
                    float by = (sigmoid(ty) + h) * stride;
                    float bw = std::exp(tw) * anchors_[level][a][0];
                    float bh = std::exp(th) * anchors_[level][a][1];

                    float x1 = bx - bw / 2.0f;
                    float y1 = by - bh / 2.0f;
                    float x2 = bx + bw / 2.0f;
                    float y2 = by + bh / 2.0f;

                    boxes.emplace_back(x1, y1, x2, y2, score, max_cls_id);
                }
            }
        }
    }

    return boxes;
}

} // namespace yolo_onnx