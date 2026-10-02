#include "process/postprocess/postprocess.hpp"
#include "process/postprocess/postprocess_core.hpp"

namespace yolo_onnx {

// ============================================================
// PostProcessPose — 姿态 / 关键点
// ============================================================
// 流程：decode（框+关键点）→ 按框过滤/还原/NMS → 框与关键点同步还原
// 关键点与框按候选下标绑定，因此用 select_indices() 取保留项的**下标**，
// 再按下标回取对应关键点。
InferResult PostProcessPose::forward(const TensorSet& outputs,
                                     const LetterboxInfo& lb) const {
    auto cands = decoder_->decode_pose(outputs, ctx_);
    if (cands.empty()) return PoseResult{};

    std::vector<Box> boxes;
    boxes.reserve(cands.size());
    for (const auto& c : cands) boxes.push_back(c.box);

    auto keep = select_indices(boxes, lb, params_);
    if (keep.empty()) return PoseResult{};

    PoseResult result;
    result.boxes.reserve(keep.size());
    result.keypoints.reserve(keep.size());
    for (int i : keep) {
        // candidates 在模型输入坐标系，需还原到原图坐标
        Box b = cands[i].box;
        scale_box(b, lb);
        result.boxes.push_back(b);

        auto kps = cands[i].extra;
        restore_keypoints(kps, lb);   // 关键点同样需要还原
        result.keypoints.push_back(std::move(kps));
    }
    return result;
}

} // namespace yolo_onnx