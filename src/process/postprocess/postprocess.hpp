#pragma once
#include "yolo_onnx/yolo_onnx_types.hpp"
#include "process/postprocess/decoder/decoder.hpp"

#include <memory>

namespace yolo_onnx {

// ============================================================
// PostProcess — 按任务分派的后处理模块
// ============================================================
// 职责：把后端原始输出（TensorSet）解码为最终结果。
//       decode（模型格式相关，委托 Decoder）→ 坐标还原 → 置信度过滤 → NMS → 截断
//
// 与 PreProcess 的对称关系：
//   PreProcess(params).run(image) → PreProcessResult{blob, letterbox}
//   PostProcess(params, decoder).forward(outputs, letterbox) → InferResult
//
// 两种使用方式：
//   1) 经由本模块：create_postprocess(task, model_type, params) 一步到位
//   2) 单独复用后处理算子：直接用 postprocess_core.hpp 里的自由函数
//      （零依赖，可整份拷到其他项目）
//
// 各任务的实现分别在 detect.cpp / segment.cpp / pose.cpp / obb.cpp，
// 按模型输出格式的解码器在 decoder/ 子目录。
// ============================================================

class PostProcess {
public:
    PostProcess(PostProcessParams params, std::shared_ptr<Decoder> decoder)
        : params_(params), decoder_(std::move(decoder)) {}

    virtual ~PostProcess() = default;

    /// 标准入口：原始输出 → 最终结果
    virtual InferResult forward(const TensorSet& outputs,
                                const LetterboxInfo& lb) const = 0;

    const PostProcessParams& params() const { return params_; }
    const std::shared_ptr<Decoder>& decoder() const { return decoder_; }

    /// 设置解码上下文（网格尺寸/类别数等，需在 forward 之前调用）
    void set_context(const DecodeContext& ctx) { ctx_ = ctx; }

protected:
    PostProcessParams        params_;
    std::shared_ptr<Decoder> decoder_;
    DecodeContext            ctx_;   // 解码上下文
};

// ============================================================
// 任务子类
// ============================================================

/// 检测：候选框 → 还原坐标 → 过滤 → NMS
class PostProcessDet : public PostProcess {
public:
    using PostProcess::PostProcess;
    InferResult forward(const TensorSet& outputs, const LetterboxInfo& lb) const override;
};

/// 分割：候选框 + mask 系数 → NMS → 由 proto mask 合成实例掩码
class PostProcessSegment : public PostProcess {
public:
    using PostProcess::PostProcess;
    InferResult forward(const TensorSet& outputs, const LetterboxInfo& lb) const override;

private:
    /// 在输出张量中定位 proto mask（通道数为 32 的那个 4D 张量）
    static int find_proto_index(const TensorSet& outputs);

    /// mask = sigmoid(proto @ coeffs)，裁剪到框区域后缩放到原图尺寸
    static std::vector<Mask> compose_masks(const TensorSet& outputs, int proto_idx,
                                           const std::vector<std::array<float, 32>>& coeffs,
                                           const LetterboxInfo& lb,
                                           const std::vector<Box>& boxes,
                                           int input_width);
};

/// 姿态：候选框 + 关键点 → NMS（框与关键点同步还原）
class PostProcessPose : public PostProcess {
public:
    using PostProcess::PostProcess;
    InferResult forward(const TensorSet& outputs, const LetterboxInfo& lb) const override;
};

/// 旋转框：候选框 → 还原坐标 → 过滤 → 旋转 IoU NMS
class PostProcessOBB : public PostProcess {
public:
    using PostProcess::PostProcess;
    InferResult forward(const TensorSet& outputs, const LetterboxInfo& lb) const override;
};

/// 按任务 + 模型类型创建后处理器（自动装配好对应 Decoder）
std::shared_ptr<PostProcess> create_postprocess(TaskType task, ModelType model_type,
                                                const PostProcessParams& params);

} // namespace yolo_onnx