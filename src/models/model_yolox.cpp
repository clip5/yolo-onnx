#include "yolo_onnx/models/model_yolox.hpp"
#include <iostream>

namespace yolo_onnx {

BoxArray ModelYOLOX::infer(const cv::Mat& image) {
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
        std::cerr << "[ModelYOLOX] Inference failed" << std::endl;
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

BoxArray ModelYOLOX::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    BoxArray boxes;
    if (output_data.empty()) return boxes;

    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;

    // YOLOX outputs: 3 tensors [1, 5+C, H, W] at strides 8, 16, 32
    // Or sometimes 1 concatenated output [1, 5+C, 8400]
    // Handle both cases

    // Case 1: Single output (concatenated)
    if (output_data.size() == 1) {
        const auto& data = output_data[0];
        const auto& shape = output_shapes[0];

        if (shape.size() != 3) {
            std::cerr << "[ModelYOLOX] Unexpected single output shape: " << shape.size() << "D" << std::endl;
            return boxes;
        }

        int channels = (int)shape[1];
        int num_grids = (int)shape[2];  // 8400 = 80*80 + 40*40 + 20*20

        // Compute grid offsets for each stride level
        // 80*80 = 6400, 40*40 = 1600, 20*20 = 400
        int grid_counts[3] = {6400, 1600, 400};
        int grid_sizes[3] = {80, 40, 20};

        int grid_offset = 0;
        for (int level = 0; level < 3; level++) {
            int stride = strides_[level];
            int grid_h = grid_sizes[level];
            int grid_w = grid_sizes[level];

            for (int g = 0; g < grid_counts[level]; g++) {
                int idx = grid_offset + g;
                int gi = idx % grid_w;
                int gj = idx / grid_w;

                // YOLOX: [tx, ty, tw, th, obj, cls0, cls1, ...]
                float obj = sigmoid(data[4 + idx * channels]);  // NCHW vs NCHW... careful

                // Actually, for concatenated output, the layout is typically:
                // [batch, channels, num_boxes] = NCHW where num_boxes = total grid cells
                // So for box i, we access data[0 + i*channels], data[1 + i*channels], etc.
                // Wait, that's wrong. For NCHW with shape [1, C, N], the layout is:
                // data[0..N-1] = channel 0, data[N..2N-1] = channel 1, etc.
                // So for box i, channel c: data[c * N + i]

                float obj_val = data[4 * num_grids + idx];  // obj channel
                float obj_conf = sigmoid(obj_val);

                // Class scores
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    float cls = sigmoid(data[(5 + c) * num_grids + idx]);
                    if (cls > max_cls) {
                        max_cls = cls;
                        max_cls_id = c;
                    }
                }

                float score = obj_conf * max_cls;
                if (score < score_thresh) continue;

                // Decode box
                float tx = data[0 * num_grids + idx];
                float ty = data[1 * num_grids + idx];
                float tw = data[2 * num_grids + idx];
                float th = data[3 * num_grids + idx];

                float bx = (sigmoid(tx) + gi) * stride;
                float by = (sigmoid(ty) + gj) * stride;
                float bw = std::exp(tw) * stride;
                float bh = std::exp(th) * stride;

                float x1 = bx - bw / 2.0f;
                float y1 = by - bh / 2.0f;
                float x2 = bx + bw / 2.0f;
                float y2 = by + bh / 2.0f;

                boxes.emplace_back(x1, y1, x2, y2, score, max_cls_id);
            }
            grid_offset += grid_counts[level];
        }
    }
    // Case 2: 3 separate outputs
    else if (output_data.size() >= 3) {
        for (int level = 0; level < 3; level++) {
            const auto& data = output_data[level];
            const auto& shape = output_shapes[level];

            if (shape.size() != 4) continue;
            int channels = (int)shape[1];
            int height   = (int)shape[2];
            int width    = (int)shape[3];
            int stride   = strides_[level];

            if (channels != 5 + num_classes) continue;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float obj = sigmoid(data[4 + h * width + w]);

                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float cls = sigmoid(data[5 + c + h * width + w]);
                        if (cls > max_cls) {
                            max_cls = cls;
                            max_cls_id = c;
                        }
                    }

                    float score = obj * max_cls;
                    if (score < score_thresh) continue;

                    float tx = data[0 + h * width + w];
                    float ty = data[1 + h * width + w];
                    float tw = data[2 + h * width + w];
                    float th = data[3 + h * width + w];

                    float bx = (sigmoid(tx) + w) * stride;
                    float by = (sigmoid(ty) + h) * stride;
                    float bw = std::exp(tw) * stride;
                    float bh = std::exp(th) * stride;

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