#pragma once
#include "yolo_onnx/model.hpp"

namespace yolo_onnx {

/// YOLOv8-pose — Pose/keypoint estimation model
/// Output: 1 tensor [1, 4+num_classes+num_keypoints*3, num_boxes]
/// Each keypoint: (x, y, visibility)  — visibility: 0=unlabeled, 1=not visible, 2=visible
class ModelV8Pose : public Model {
public:
    InferResult infer(const cv::Mat& image) override;

    /// Full pose inference: returns boxes + keypoints
    PoseResult infer_pose(const cv::Mat& image);

protected:
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int strides_[3] = {8, 16, 32};

    /// Decode output into candidate boxes and their keypoints
    struct DecodedPoseBox {
        Box box;
        std::vector<Keypoint> keypoints;
    };

    std::vector<DecodedPoseBox> decode_pose_candidates(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const;
};

} // namespace yolo_onnx