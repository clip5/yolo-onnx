#pragma once
#include "yolo_onnx/backend.hpp"
#include <onnxruntime_cxx_api.h>

namespace yolo_onnx {

/// ONNX Runtime backend (CPU)
class OnnxruntimeBackend : public Backend {
public:
    OnnxruntimeBackend();
    ~OnnxruntimeBackend() override;

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

    std::string name() const override { return "onnxruntime"; }

private:
    Ort::Env                    env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "yolo-onnx"};
    Ort::SessionOptions         session_options_;
    Ort::Session                session_{nullptr};
    Ort::MemoryInfo             memory_info_{nullptr};

    std::vector<std::string>          input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<ONNXTensorElementDataType> input_types_;

    std::vector<std::string>          output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<ONNXTensorElementDataType> output_types_;

    bool loaded_{false};
};

} // namespace yolo_onnx