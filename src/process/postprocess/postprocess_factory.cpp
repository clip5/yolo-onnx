#include "process/postprocess/postprocess.hpp"

#include <iostream>

namespace yolo_onnx {

// ============================================================
// 工厂：按「任务 × 模型格式」装配后处理器
// ============================================================
// 两个正交轴在这里汇合：
//   TaskType    → 决定用哪个 PostProcess 子类（输出什么结果）
//   ModelType   → 决定用哪个 Decoder（怎么解原始张量）
std::shared_ptr<PostProcess> create_postprocess(TaskType task, ModelType model_type,
                                                const PostProcessParams& params) {
    auto decoder = create_decoder(model_type);
    if (!decoder) {
        std::cerr << "[create_postprocess] No decoder for model type: "
                  << model_type_name(model_type) << std::endl;
        return nullptr;
    }

    switch (task) {
        case TaskType::Detect:
            return std::make_shared<PostProcessDet>(params, decoder);
        case TaskType::Segment:
            return std::make_shared<PostProcessSegment>(params, decoder);
        case TaskType::Pose:
            return std::make_shared<PostProcessPose>(params, decoder);
        case TaskType::OBB:
            return std::make_shared<PostProcessOBB>(params, decoder);
        default:
            std::cerr << "[create_postprocess] Unknown task type: "
                      << task_type_name(task) << std::endl;
            return nullptr;
    }
}

} // namespace yolo_onnx