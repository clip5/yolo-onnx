#include "process/postprocess/postprocess.hpp"
#include "process/postprocess/postprocess_core.hpp"

namespace yolo_onnx {

// ============================================================
// PostProcessDet — 目标检测
// ============================================================
// 流程：decode（模型格式相关）→ 过滤 → 坐标还原 → NMS → 截断
InferResult PostProcessDet::forward(const TensorSet& outputs,
                                    const LetterboxInfo& lb) const {
    BoxArray candidates = decoder_->decode_detect(outputs, ctx_);
    if (candidates.empty()) return DetectResult{};

    // detect_pipeline 内部完成：过滤 → 还原 → NMS → 截断
    return DetectResult{ detect_pipeline(std::move(candidates), lb, params_) };
}

} // namespace yolo_onnx