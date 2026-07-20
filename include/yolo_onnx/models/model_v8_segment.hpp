#pragma once
#include "yolo_onnx/model.hpp"

namespace yolo_onnx {

/// YOLOv8-seg — Instance segmentation model
/// Two outputs:
///   1. Detection + mask coeffs: [1, 4+num_classes+32, num_boxes]
///   2. Proto masks:             [1, 32, H, W]  (H,W = input/4)
class ModelV8Segment : public Model {
public:
    InferResult infer(const cv::Mat& image) override;

    /// Full segment inference: returns boxes + masks
    SegmentResult infer_segment(const cv::Mat& image);

protected:
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int strides_[3] = {8, 16, 32};
    static constexpr int num_masks_ = 32;  // number of mask coefficient channels

    /// Decode output into candidate boxes and their mask coefficients
    struct DecodedBox {
        Box box;
        float mask_coeffs[32];
    };
    std::vector<DecodedBox> decode_candidates(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const;

    /// Compute final masks from proto masks and mask coefficients
    std::vector<Mask> compute_masks(
        const std::vector<float>&   proto_data,
        int                         proto_h,
        int                         proto_w,
        const std::vector<DecodedBox>& candidates,
        const LetterboxInfo&        letterbox,
        int                         orig_w,
        int                         orig_h
    ) const;
};

} // namespace yolo_onnx