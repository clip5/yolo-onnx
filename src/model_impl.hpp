#pragma once
// ============================================================
// model_impl.hpp — Model 的内部实现（不对外暴露）
// ============================================================
// Model 的成员与内部方法集中定义在此，公共头 yolo_onnx.hpp 只暴露
// Config + 5 个公开方法，实现细节（backend / preprocess / postprocess
// 等成员，以及 protected 虚函数）通过下面的 Impl 结构隐藏。
//
// 这样做的目的：对外接口稳定 —— 新增内部缓存、拆分流水线、增加后处理
// 阶段都不会改变 Model 的公开 API，也不会把内部类型泄漏给使用者。

#include "yolo_onnx/yolo_onnx.hpp"

#include <memory>

namespace yolo_onnx {

class Backend;
class PreProcess;
class PostProcess;

namespace detail {

/// Model 的全部内部状态与内部方法
struct ModelImpl {
    // ---- 内部状态 ----
    std::shared_ptr<Backend>     backend;
    mutable std::shared_ptr<PreProcess>  preprocessor;  // 复用画布/通道缓冲
    std::shared_ptr<PostProcess> postprocess;

    // ---- 内部方法 ----

    /// Letterbox resize + normalize（复用独立的 PreProcess 模块，按尺寸懒初始化）
    /// @param model_type 决定预处理差异（YOLOX 用 ImageNet mean/std，PPYOLOE 保持 BGR）
    PreProcessResult preprocess(const cv::Mat& image, ModelType model_type,
                                int target_w, int target_h) const;

    /// 预处理 → 前向 → 后处理 的统一流程。
    /// 解码 / 坐标还原 / 过滤 / NMS 全部在 PostProcess 内部完成，
    /// 本函数不含任何与具体模型输出格式相关的解码代码。
    InferResult run_pipeline(const cv::Mat& image, const Model::Config& cfg);
};

} // namespace detail

} // namespace yolo_onnx