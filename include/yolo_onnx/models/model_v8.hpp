#pragma once
#include "yolo_onnx/model.hpp"

namespace yolo_onnx {

/// YOLOv8 / YOLOv11 / YOLO26 — Anchor-free detector (single output)
/// Output: 1 tensor [1, 4+C, num_boxes]  (num_boxes = sum of all grid cells)
/// Box format: cx, cy, w, h in grid space (need stride scaling)
class ModelV8 : public Model {
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