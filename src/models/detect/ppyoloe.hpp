#pragma once
#include "yolo_onnx/yolo_onnx.hpp"

namespace yolo_onnx {

/// PPYOLOE — Anchor-free detector (Baidu PaddleDetection)
/// Output: 6 tensors (3 cls + 3 reg) at strides 8, 16, 32
///   cls[i]: [1, C, H, W]
///   reg[i]: [1, 4, H, W]
class ModelPPYOLOE : public Model {
public:
    InferResult infer(const cv::Mat& image) override;
protected:
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int strides_[3] = {8, 16, 32};
};

} // namespace yolo_onnx