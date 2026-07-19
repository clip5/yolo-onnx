#pragma once
#include "yolo_onnx/backend.hpp"
#include <vector>
#include <string>
#include <memory>

namespace yolo_onnx {

/// CANN backend (Huawei Ascend NPU)
/// Loads .om model files converted from ONNX via atc tool
/// Uses AscendCL (acl) API
class CANNBackend : public Backend {
public:
    CANNBackend();
    ~CANNBackend() override;

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

    std::string name() const override { return "cann"; }

private:
    // AscendCL handles (opaque pointers)
    void* context_{nullptr};    // aclrtContext
    void* stream_{nullptr};     // aclrtStream
    void* model_id_{nullptr};   // aclmdlDesc (model ID as uint32_t, stored as pointer)

    // Model info
    uint32_t model_id_{0};
    size_t   input_size_{0};
    size_t   output_size_{0};
    void*    input_buffer_{nullptr};
    void*    output_buffer_{nullptr};

    std::vector<std::string>          input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::string>          output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;

    int  device_id_{0};
    bool loaded_{false};

    /// Initialize AscendCL
    bool init_acl();
    /// Destroy AscendCL resources
    void destroy_acl();
};

} // namespace yolo_onnx