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
    /// @return blob 为 NCHW cv::Mat（dims=4, size={1,3,H,W}，连续），可直接交推理后端；
    ///         letterbox 携带坐标还原所需的 scale/padding 信息。
    /// @note  返回的 blob 独立持有内存（不复用内部缓冲），可安全跨多次调用持有。
    virtual PreProcessResult run(const cv::Mat& image) const;

    const PreProcessParams& params() const { return params_; }

protected:
    PreProcessParams params_;

    /// 中间缓冲复用：letterbox 画布与通道分离结果不逃逸出本次调用，可安全复用
    mutable cv::Mat           canvas_;
    mutable std::vector<cv::Mat> channels_;
};

} // namespace yolo_onnx
