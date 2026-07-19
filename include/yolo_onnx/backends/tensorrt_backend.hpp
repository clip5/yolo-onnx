#pragma once
#include "yolo_onnx/backend.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>

namespace yolo_onnx {

/// TensorRT backend (NVIDIA GPU)
/// Supports .engine files (pre-built) and .onnx files (built at runtime)
class TensorRTBackend : public Backend {
public:
    TensorRTBackend();
    ~TensorRTBackend() override;

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

    std::string name() const override { return "tensorrt"; }

private:
    // RAII wrapper for TensorRT logger
    class TRTLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    TRTLogger                  logger_;
    std::shared_ptr<nvinfer1::IRuntime>      runtime_{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine>   engine_{nullptr};
    std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    // Device buffers
    void* device_buffers_[2]{};   // input + output (max 2 I/O tensors for typical YOLO)
    std::vector<void*> device_output_ptrs_;
    std::vector<size_t> output_sizes_;

    // Host-side info
    std::vector<std::string>          input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::string>          output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;

    int  device_id_{0};
    bool loaded_{false};

    /// Build engine from ONNX file (uses nvinfer1::IBuilder)
    bool build_from_onnx(const std::string& onnx_path, bool enable_fp16, bool enable_int8);

    /// Load pre-built engine from file
    bool load_engine(const std::string& engine_path);

    /// Save engine to file
    bool save_engine(const std::string& engine_path);
};

} // namespace yolo_onnx