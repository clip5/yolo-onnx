#pragma once
#include "yolo_onnx/yolo_onnx_types.hpp"
#include "process/postprocess/decoder.hpp"

#include <memory>
#include <vector>

namespace yolo_onnx {

// ============================================================
// PostProcess — 统一的后处理接口（按任务分派）
// ============================================================
// 设计目标：可脱离推理流程单独使用，也可在 Model 中一行调用。
//
//   TensorSet（后端原始输出）
//     → Decoder（模型格式相关，可插拔）→ 候选框（模型输入坐标系）
//     → PostProcess（本模块，任务相关）→ 结果（原图坐标系）
//
// 标准入口：
//   PostProcess pp(params, decoder);
//   InferResult r = pp.forward(outputs, letterbox);
//
// Decoder 与 PostProcess 正交：Decoder 按「模型格式」组织（v5/v8/yolox/ppyoloe），
// PostProcess 按「任务」组织（Det/Segment/Pose/OBB）。新增模型格式只加 Decoder；
// 新增任务只加一个 PostProcess 子类。
//
// 组件级入口（需要自定义流程时单独调用）：
//   filter_by_score() / suppress() / restore_boxes() ...
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

    // ============================================================
    // 组件级入口（模型无关，各子类共用）
    // ============================================================

    /// 置信度过滤
    BoxArray    filter_by_score(const BoxArray& boxes) const;
    OBBBoxArray filter_by_score(const OBBBoxArray& boxes) const;

    /// NMS：返回保留下来的框
    BoxArray    suppress(const BoxArray& boxes) const;
    OBBBoxArray suppress_rotated(const OBBBoxArray& boxes) const;

    /// 坐标还原（letterbox 逆映射），in-place；不改动 score / label
    static void restore_boxes(BoxArray& boxes, const LetterboxInfo& lb);
    static void restore_keypoints(std::vector<Keypoint>& kps, const LetterboxInfo& lb);
    static void restore_obb(OBBBoxArray& boxes, const LetterboxInfo& lb);

    /// 在一批候选框中做「过滤 → 还原 → NMS」，返回保留的原始下标。
    /// segment/pose 需要按 box 下标回取附属信息（mask 系数/ 关键点），
    /// 因此必须保留索引，不能只返回框本身。
    std::vector<int> select(const std::vector<Box>& candidates,
                            const LetterboxInfo& lb) const;

    /// 保留分数最高的 max_detections 个（0 表示不限制）
    static BoxArray    top_k(BoxArray boxes, int max_detections);
    static OBBBoxArray top_k(OBBBoxArray boxes, int max_detections);

protected:
    PostProcessParams          params_;
    std::shared_ptr<Decoder>   decoder_;

    /// 由 Model::make_decode_context() 构造后传入
    DecodeContext              ctx_;

public:
    /// 设置解码上下文（模型配置相关），需在 forward 之前调用
    void set_context(const DecodeContext& ctx) { ctx_ = ctx; }
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

    /// mask = sigmoid(proto @ coeffs)，裁剪到框区域后缩放到原图尺寸。
    /// @param input_width 模型输入宽度（用于 proto mask 空间换算）
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

// ============================================================
// 自由函数：基础工具（对外 API，Python 绑定直接使用）
// ============================================================

/// 两框 IoU
float iou(const Box& a, const Box& b);

/// CPU NMS（轴对齐框），返回保留框的索引
std::vector<int> nms(const BoxArray& boxes, float iou_threshold);

/// OBB 旋转 IoU 与 NMS
float obb_iou(const OBBBox& a, const OBBBox& b);
std::vector<int> obb_nms(const OBBBoxArray& boxes, float iou_threshold);

} // namespace yolo_onnx