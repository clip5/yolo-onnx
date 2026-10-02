#pragma once
#include "core/backend.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

namespace yolo_onnx {

/// ONNX Runtime backend — supports any hardware via Execution Providers
///
/// Execution Provider 通过 Config::custom_config 指定，格式：
///   ep=cuda                — 使用 CUDA EP
///   ep=tensorrt            — 使用 TensorRT EP
///   ep=cuda;device=1       — CUDA EP + 指定设备
///   不指定则默认 CPU
///
/// 注意：需要安装对应版本的 onnxruntime 库（如 onnxruntime-gpu 才支持 CUDA EP）
class OnnxruntimeBackend : public Backend {
public:
    OnnxruntimeBackend();
    ~OnnxruntimeBackend() override;

    bool load(const Config& config) override;

    bool forward(
        const std::vector<std::string>&          input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<cv::Mat>&             input_data,
        const std::vector<std::string>&          output_names,
        TensorSet&                               outputs
    ) override;

    std::vector<std::string>          get_input_names()  const override;
    std::vector<std::vector<int64_t>> get_input_shapes() const override;
    std::vector<std::string>          get_output_names()  const override;
    std::vector<std::vector<int64_t>> get_output_shapes() const override;

    std::string name() const override { return name_; }

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

    std::string name_{"onnxruntime"};   // backend name, includes EP suffix
    bool loaded_{false};

    /// Parse custom_config and register execution providers
    bool setup_execution_providers(const std::string& custom_config, int device_id);
};

} // namespace yolo_onnx