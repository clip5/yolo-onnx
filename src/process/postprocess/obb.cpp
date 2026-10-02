#include "process/postprocess/postprocess.hpp"
#include "process/postprocess/postprocess_core.hpp"

namespace yolo_onnx {

// ============================================================
// PostProcessOBB — 旋转框检测
// ============================================================
// 流程：decode（框+角度）→ 过滤 → 坐标还原 → 旋转 IoU NMS → 截断
// OBB 无附属信息，不需要下标映射，直接用 obb_pipeline 即可。
InferResult PostProcessOBB::forward(const TensorSet& outputs,
                                    const LetterboxInfo& lb) const {
    auto candidates = decoder_->decode_obb(outputs, ctx_);
    if (candidates.empty()) return OBBResult{};

    return OBBResult{ obb_pipeline(std::move(candidates), lb, params_) };
}

} // namespace yolo_onnx