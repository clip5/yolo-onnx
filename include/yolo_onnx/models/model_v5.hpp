#pragma once
#include "yolo_onnx/model.hpp"

namespace yolo_onnx {

/// YOLOv5 — Anchor-based detector
/// Output: 3 tensors [1, 3*(5+C), H, W] at strides 8, 16, 32
class ModelV5 : public Model {
public:
    InferResult infer(const cv::Mat& image) override;
protected:
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int anchors_[3][3][2] = {
        {{10, 13}, {16, 30}, {33, 23}},   // P3 / stride 8
        {{30, 61}, {62, 45}, {59, 119}},  // P4 / stride 16
        {{116, 90}, {156, 198}, {373, 326}} // P5 / stride 32
    };
    static constexpr int strides_[3] = {8, 16, 32};
};

} // namespace yolo_onnx