#pragma once
#include "yolo_onnx/yolo_onnx.hpp"

namespace yolo_onnx {

/// YOLOX — Anchor-free detector
/// Output: 3 tensors [1, (5+C), H, W] at strides 8, 16, 32
class ModelYOLOX : public Model {
public:
    InferResult infer(const cv::Mat& image) override;
protected:
    /// YOLOX expects ImageNet mean/std normalization
    PreProcessParams make_preprocess_params() const override {
        auto p = PreProcessParams::for_model(config_.model_type,
                                             config_.input_width, config_.input_height);
        static const float mean[3] = {0.485f, 0.456f, 0.406f};
        static const float std_[3] = {0.229f, 0.224f, 0.225f};
        for (int c = 0; c < 3; c++) { p.mean[c] = mean[c]; p.std[c] = std_[c]; }
        return p;
    }

    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;

private:
    static constexpr int strides_[3] = {8, 16, 32};
};

} // namespace yolo_onnx