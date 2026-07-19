#pragma once
#include "yolo_onnx/backend.hpp"
#include <vector>
#include <string>
#include <cstdint>

// Forward declare RKNN types (include rknn_api.h in .cpp)
struct rknn_context;

namespace yolo_onnx {

/// RKNN backend (Rockchip NPU)
/// Loads .rknn model files converted from ONNX via rknn-toolkit
class RKNNBackend : public Backend {
public:
    RKNNBackend();
    ~RKNNBackend() override;

    bool load(const Config& config) override;

    bool forward(
        const std::vector<std::string>&          input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<float>&                input_data,
        const std::vector<std::string>&          output_names,
        std::vector<std::vector<int64_t>>&       output_shapes,
        std::vector<std::vector<float>>&         output_data
    ) override;

    std::vector<std::string>          get_input_names()  const override;
    std::vector<std::vector<int64_t>> get_input_shapes() const override;
    std::vector<std::string>          get_output_names()  const override;
    std::vector<std::vector<int64_t>> get_output_shapes() const override;

    std::string name() const override { return "rknn"; }

private:
    // RKNN context (opaque pointer to rknn_context)
    rknn_context* ctx_{nullptr};

    // I/O tensor info (cached at load time)
    std::vector<std::string>          input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::string>          output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Runtime info for zero-copy inference
    int num_inputs_{0};
    int num_outputs_{0};

    bool loaded_{false};
};

} // namespace yolo_onnx