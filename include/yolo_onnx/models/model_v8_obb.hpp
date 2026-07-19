#pragma once
#include "yolo_onnx/model.hpp"

namespace yolo_onnx {

/// YOLOv8-obb — Oriented bounding box detection
/// Output: 1 tensor [1, 4+num_classes+1, num_boxes]  (extra 1 = angle)
/// Box format: cx, cy, w, h (grid space), angle (radians)
class ModelV8OBB : public Model {
public:
    BoxArray infer(const cv::Mat& image) override;

    /// Full OBB inference: returns oriented bounding boxes
    OBBResult infer_obb(const cv::Mat& image);

protected:
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int strides_[3] = {8, 16, 32};

    /// Decode output into OBB candidates
    OBBBoxArray decode_obb_candidates(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const;
};

} // namespace yolo_onnx