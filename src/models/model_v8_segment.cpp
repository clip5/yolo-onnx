#include "yolo_onnx/models/model_v8_segment.hpp"
#include "yolo_onnx/backend.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace yolo_onnx {

InferResult ModelV8Segment::infer(const cv::Mat& image) {
    return infer_segment(image);
}

SegmentResult ModelV8Segment::infer_segment(const cv::Mat& image) {
    SegmentResult result;

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
        std::cerr << "[ModelV8Segment] Inference failed" << std::endl;
        return result;
    }

    // 4. Decode candidates (boxes + mask coefficients)
    auto candidates = decode_candidates(output_data, output_shapes);

    // 5. Scale boxes back to original image
    for (auto& cand : candidates) {
        scale_box(cand.box, pre.letterbox);
    }

    // 6. NMS on boxes
    BoxArray boxes;
    for (const auto& cand : candidates) {
        boxes.push_back(cand.box);
    }
    auto keep = nms(boxes, config_.nms_thresh);

    // 7. Collect NMS-kept candidates
    std::vector<DecodedBox> kept_candidates;
    BoxArray kept_boxes;
    for (int idx : keep) {
        kept_candidates.push_back(candidates[idx]);
        kept_boxes.push_back(boxes[idx]);
    }

    // 8. Compute masks (if proto mask output is available)
    // Proto masks are typically output_data[1] (or the last output)
    // YOLOv8-seg outputs: [detection, proto_masks]
    // or sometimes [proto_masks, detection] — check which tensor has 32 channels
    int proto_idx = -1;
    for (int i = 0; i < (int)output_data.size(); i++) {
        if (output_shapes[i].size() == 4) {
            auto& shape = output_shapes[i];
            // Proto mask: [1, 32, H, W] where H, W are typically input/4
            if (shape[1] == 32 && shape[2] > 0 && shape[3] > 0) {
                proto_idx = i;
                break;
            }
        }
    }

    if (proto_idx >= 0 && !kept_candidates.empty()) {
        auto& proto_shape = output_shapes[proto_idx];
        int proto_h = (int)proto_shape[2];
        int proto_w = (int)proto_shape[3];

        auto masks = compute_masks(
            output_data[proto_idx], proto_h, proto_w,
            kept_candidates, pre.letterbox,
            pre.letterbox.orig_w, pre.letterbox.orig_h
        );

        result.boxes = kept_boxes;
        result.masks = masks;
    } else {
        result.boxes = kept_boxes;
    }

    return result;
}

BoxArray ModelV8Segment::decode_output(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    auto candidates = decode_candidates(output_data, output_shapes);
    BoxArray boxes;
    for (const auto& cand : candidates) {
        boxes.push_back(cand.box);
    }
    return boxes;
}

std::vector<ModelV8Segment::DecodedBox> ModelV8Segment::decode_candidates(
    const std::vector<std::vector<float>>&   output_data,
    const std::vector<std::vector<int64_t>>& output_shapes
) const {
    std::vector<DecodedBox> candidates;
    if (output_data.empty()) return candidates;

    int num_classes = config_.num_classes;
    float score_thresh = config_.score_thresh;

    // YOLOv8-seg: output[0] = detection [1, 4+num_classes+32, num_boxes]
    // output[1] = proto masks [1, 32, H, W]
    const auto& data = output_data[0];
    const auto& shape = output_shapes[0];

    if (shape.size() < 3) return candidates;

    // Try to determine the layout
    // Case: [1, C, N] or [1, N, C] or [1, C, H, W]
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

    int cls_start = 4;
    int coeff_start = 4 + num_classes;

    if (is_chw_layout) {
        // 4D output: [1, C, H, W] — treat as grids per stride
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

                // Decode box (cx, cy, w, h in grid space)
                float cx = data[0 * height * width + h * width + w];
                float cy = data[1 * height * width + h * width + w];
                float bw = data[2 * height * width + h * width + w];
                float bh = data[3 * height * width + h * width + w];

                // Assume raw values needing sigmoid + grid offset
                // Use the input size to determine stride
                int stride = config_.input_width / width;
                float bx = (sigmoid(cx) + w) * stride;
                float by = (sigmoid(cy) + h) * stride;
                float box_w = bw * stride;
                float box_h = bh * stride;

                float x1 = bx - box_w / 2.0f;
                float y1 = by - box_h / 2.0f;
                float x2 = bx + box_w / 2.0f;
                float y2 = by + box_h / 2.0f;

                DecodedBox cand;
                cand.box = Box(x1, y1, x2, y2, max_cls, max_cls_id);

                // Read mask coefficients
                for (int m = 0; m < num_masks_; m++) {
                    cand.mask_coeffs[m] = data[(coeff_start + m) * height * width + h * width + w];
                }
                candidates.push_back(cand);
            }
        }
    } else {
        // 3D output [1, C, N] — concatenated grid
        // Build grid offsets: 80*80 + 40*40 + 20*20 = 8400
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

                // Check if values are raw (0-1) or already decoded
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

                DecodedBox cand;
                cand.box = Box(x1, y1, x2, y2, max_cls, max_cls_id);

                // Read mask coefficients
                for (int m = 0; m < num_masks_; m++) {
                    cand.mask_coeffs[m] = data[(coeff_start + m) * num_boxes + idx];
                }
                candidates.push_back(cand);
            }
            grid_offset += grid_counts[level];
        }
    }

    return candidates;
}

std::vector<Mask> ModelV8Segment::compute_masks(
    const std::vector<float>&   proto_data,
    int                         proto_h,
    int                         proto_w,
    const std::vector<DecodedBox>& candidates,
    const LetterboxInfo&        letterbox,
    int                         orig_w,
    int                         orig_h
) const {
    std::vector<Mask> masks;

    for (const auto& cand : candidates) {
        // Compute mask: sigmoid(proto_masks @ coeffs)
        // proto_data: [1, 32, proto_h, proto_w] in CHW format
        // For each pixel (y, x) in proto mask:
        //   mask_val = sum(proto_data[c, y, x] * coeffs[c]) for c in 0..31
        // Then sigmoid

        Mask mask(proto_w, proto_h);
        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                float val = 0.0f;
                for (int c = 0; c < num_masks_; c++) {
                    val += proto_data[c * proto_h * proto_w + h * proto_w + w] * cand.mask_coeffs[c];
                }
                mask.data[h * proto_w + w] = sigmoid(val);
            }
        }

        // Crop mask to the box region (in proto mask space)
        // Box coordinates are in model output space (before scale_box)
        // We need to map the box to proto mask space
        // The proto mask is typically at 1/4 the input resolution
        float proto_scale = (float)proto_w / config_.input_width;
        Box box = cand.box;  // box in model output space

        int x1 = std::max(0, (int)((box.x1 - letterbox.pad_left) * proto_scale));
        int y1 = std::max(0, (int)((box.y1 - letterbox.pad_top) * proto_scale));
        int x2 = std::min(proto_w - 1, (int)((box.x2 - letterbox.pad_left) * proto_scale));
        int y2 = std::min(proto_h - 1, (int)((box.y2 - letterbox.pad_top) * proto_scale));

        // Crop: zero out everything outside the box
        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                if (w < x1 || w > x2 || h < y1 || h > y2) {
                    mask.data[h * proto_w + w] = 0.0f;
                }
            }
        }

        // Resize mask to original image size
        cv::Mat mask_cv(proto_h, proto_w, CV_32FC1, mask.data.data());
        cv::Mat mask_resized;
        cv::resize(mask_cv, mask_resized, cv::Size(orig_w, orig_h));

        // Threshold mask
        cv::Mat mask_thresh;
        cv::threshold(mask_resized, mask_thresh, 0.5f, 1.0f, cv::THRESH_BINARY);

        // Convert back to vector
        Mask final_mask(orig_w, orig_h);
        for (int h = 0; h < orig_h; h++) {
            for (int w = 0; w < orig_w; w++) {
                final_mask.data[h * orig_w + w] = mask_thresh.at<float>(h, w);
            }
        }

        masks.push_back(final_mask);
    }

    return masks;
}

} // namespace yolo_onnx