#pragma once
#include "yolo_onnx/yolo_onnx_types.hpp"

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace yolo_onnx {

// ============================================================
// Inference Backend Interface
// ============================================================
// 当前只有 onnxruntime 一个后端，硬件加速通过 ONNX Runtime
// Execution Provider 实现（CUDA / TensorRT / CANN 等），
// 在 Config::custom_config 中通过 ep=xxx 指定。
// ============================================================

class Backend {
public:
    virtual ~Backend() = default;

    /// Backend configuration
    struct Config {
        std::string model_path;         // Path to model file (.onnx / .engine / .rknn / .om ...)
        int         device_id    = 0;   // GPU / NPU device ID
        int         num_threads  = 4;   // CPU threads (for onnxruntime)
        bool        enable_fp16  = false;
        bool        enable_int8  = false;
        std::string custom_config;      // Backend-specific config string (key=value;...)
    };

    /// Load model from file
    virtual bool load(const Config& config) = 0;

    /// Run inference
    /// @param input_names   input tensor names
    /// @param input_shapes  input tensor shapes (NCHW format)
    /// @param input_data    input tensor data (float32, contiguous) — one cv::Mat per input
    /// @param output_names  output tensor names
    /// @param outputs       [out] 输出张量集合（连续 float32 Mat + 各自形状）
    virtual bool forward(
        const std::vector<std::string>&    input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<cv::Mat>&       input_data,
        const std::vector<std::string>&    output_names,
        TensorSet&                         outputs
    ) = 0;

    /// Get input tensor info
    virtual std::vector<std::string>          get_input_names()  const = 0;
    virtual std::vector<std::vector<int64_t>> get_input_shapes() const = 0;

    /// Get output tensor info
    virtual std::vector<std::string>          get_output_names()  const = 0;
    virtual std::vector<std::vector<int64_t>> get_output_shapes() const = 0;

    /// Get backend name
    virtual std::string name() const = 0;
};

/// Factory function: create a backend by name
/// Supported: "onnxruntime"
std::shared_ptr<Backend> create_backend(const std::string& backend_name);

} // namespace yolo_onnx