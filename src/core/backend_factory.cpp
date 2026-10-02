#include "core/backend.hpp"
#include "core/onnxruntime_backend.hpp"

#ifdef YOLO_ONNX_WITH_OPENVINO
#include "core/openvino_backend.hpp"
#endif

#include <iostream>

namespace yolo_onnx {

// ============================================================
// Backend factory
// ============================================================
// 主线是 onnxruntime：所有硬件加速都通过 ONNX Runtime 的
// Execution Provider 实现，编译期不依赖任何 EP 库，EP 由
// Config::custom_config 的 ep= 选择（cpu/cuda/tensorrt/openvino/
// qnn/coreml/xnnpack/...），见 onnxruntime_backend.cpp 的 EP 分派。
//
// 下面是"独立运行时后端"的注册位（不走 ORT，直接调厂商 SDK）。
// 默认全部关闭：代码保留但不对外暴露，需要时用 CMake 宏打开，
// 这样主线保持轻量，不绑架未使用的 SDK 依赖。
//   YOLO_ONNX_WITH_OPENVINO → "openvino"
//     官方 ORT 发行包不含 libonnxruntime_providers_openvino.so
//     （OpenVINO EP 只随 Intel 自定义版 ORT 发布），所以想用
//     OpenVINO 就走这个独立后端，直接调 ov::Core。
//   未来可同模式增加：
//     YOLO_ONNX_WITH_CANN  → "cann"（昇腾，onnx→ATC→.om）
//     YOLO_ONNX_WITH_RKNN  → "rknn"（瑞芯微，onnx→rknn-toolkit2→.rknn）
// ============================================================

std::shared_ptr<Backend> create_backend(const std::string& backend_name) {
    if (backend_name == "onnxruntime" || backend_name.empty()) {
        return std::make_shared<OnnxruntimeBackend>();
    }

#ifdef YOLO_ONNX_WITH_OPENVINO
    if (backend_name == "openvino") {
        return std::make_shared<OpenvinoBackend>();
    }
#endif

    std::cerr << "[create_backend] Unknown backend: " << backend_name << std::endl;
    std::cerr << "[create_backend] Available: onnxruntime";
#ifdef YOLO_ONNX_WITH_OPENVINO
    std::cerr << ", openvino";
#endif
    std::cerr << std::endl;
    std::cerr << "[create_backend] Hint: hardware acceleration on onnxruntime is "
                 "selected via custom_config (ep=cuda / ep=tensorrt / ep=qnn / ...)"
              << std::endl;
    return nullptr;
}

} // namespace yolo_onnx
