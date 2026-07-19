#pragma once
#include "yolo_onnx/model.hpp"

namespace yolo_onnx {

/// YOLOX — Anchor-free detector
/// Output: 3 tensors [1, (5+C), H, W] at strides 8, 16, 32
class ModelYOLOX : public Model {
public:
    BoxArray infer(const cv::Mat& image) override;
protected:
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int strides_[3] = {8, 16, 32};
};

} // namespace yolo_onnx