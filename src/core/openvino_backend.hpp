#pragma once
#include "core/backend.hpp"

#include <memory>

namespace yolo_onnx {

/// OpenVINO backend — loads .onnx directly (no conversion step needed).
///
/// 通过 Config::custom_config 指定设备，格式：
///   device=CPU / GPU / AUTO / NPU / GPU.1 ...
///   不指定则默认 CPU
///
/// 与 onnxruntime 后端同级：吃同一个 .onnx、走同一套
/// PreProcess / Decoder / PostProcess 管线，仅推理引擎不同。
/// 由 backend_factory.cpp 在定义了 YOLO_ONNX_WITH_OPENVINO 时注册，
/// 默认不参与编译，对外不暴露——官方 ORT 发行包不含
/// libonnxruntime_providers_openvino.so（OpenVINO EP 只随 Intel
/// 自定义版 ORT 发布），要用 OpenVINO 就走这个独立后端。
/// 未来 CANN (.om, 由 ATC 从 onnx 转换) / RKNN (.rknn) 后端同样
/// 实现 Backend 接口并按同名模式在工厂中注册。
class OpenvinoBackend : public Backend {
public:
    OpenvinoBackend();
    ~OpenvinoBackend() override;

    bool load(const Config& config) override;
    bool forward(
        const std::vector<std::string>&          input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<cv::Mat>&              input_data,
        const std::vector<std::string>&          output_names,
        TensorSet&                               outputs
    ) override;

    std::vector<std::string>          get_input_names()  const override;
    std::vector<std::vector<int64_t>> get_input_shapes() const override;
    std::vector<std::string>          get_output_names()  const override;
    std::vector<std::vector<int64_t>> get_output_shapes() const override;

    std::string name() const override { return name_; }

private:
    struct Impl;                       // PIMPL，隐藏 openvino 头文件依赖
    std::unique_ptr<Impl> impl_;
    std::string name_{"openvino"};
    bool loaded_{false};

    /// Parse custom_config → OpenVINO device name
    std::string parse_device(const std::string& custom_config) const;
};

} // namespace yolo_onnx
