#pragma once
#include "yolo_onnx/yolo_onnx_types.hpp"

#include <opencv2/core.hpp>

namespace yolo_onnx {

// ============================================================
// PreProcess — 独立的预处理模块
// ============================================================
// 职责：图像 letterbox 缩放 + BGR→RGB + 归一化 + NCHW blob 打包。
// 与模型解耦：通过 PreProcessParams（见 yolo_onnx_types.hpp）描述不同
// 模型的预处理差异，新算法只需配置参数，无需改动推理流程。
// ============================================================

class PreProcess {
public:
    explicit PreProcess(const PreProcessParams& params);

    virtual ~PreProcess() = default;

    /// Letterbox + normalize + 打包为 NCHW float blob
    virtual PreProcessResult run(const cv::Mat& image) const;

    const PreProcessParams& params() const { return params_; }

protected:
    PreProcessParams params_;
};

} // namespace yolo_onnx
