#pragma once
#include "yolo_onnx/yolo_onnx_types.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace yolo_onnx {

// ============================================================
// Decoder — 模型输出格式的解码器（可插拔）
// ============================================================
// 职责：把后端输出的原始张量（TensorSet）解码成「候选框」。
//       候选框仍处于模型输入尺寸坐标系，坐标还原/过滤/NMS 由 PostProcess 负责。
//
// 边界：
//   Backend → TensorSet → Decoder（本模块，模型格式相关）→ 候选框
//           → PostProcess（模型无关）→ 最终结果
//
// 为什么要可插拔：不同导出格式（v5 anchor / v8 DFL / yolox / ppyoloe ...）
// 的解码规则完全不同，且同一代模型的不同导出方式（1 个拼接输出 / 3 个
// 分层输出 / 6 个 cls+reg 输出）也不同。做成独立类后，新增一种格式只需
// 新增一个 Decoder 并注册到 create_decoder()，不必改动已有代码或 Model。
//
// 注意：Decoder 与「任务」正交——按格式组织（V5Decoder/V8Decoder/...），
//       由对应的 PostProcess 子类（Det/Segment/Pose/OBB）调用。
//
// DecodeContext（解码上下文）定义在公共 yolo_onnx_types.hpp，与 Decoder 解耦，
// 使 postprocess/decoder 头文件可单独抽出复用而不牵动 Model。

/// 解码器基类。子类覆写自己格式所需的 decode_* 方法，未用到的返回空。
///
/// 三类任务的中间产物：box 与其附属信息（mask 系数/ 关键点）在解码阶段是
/// 绑定的（要靠同一个候选框的索引去取 proto mask 或 keypoint），因此先解出
/// 候选对象，再由 PostProcess 做还原/过滤/NMS。
class Decoder {
public:
    /// 带附属信息的候选框（segment 的 mask 系数 / pose 的关键点）
    template <typename Extra>
    struct Candidate {
        Box   box;
        Extra extra;
    };

    virtual ~Decoder() = default;

    virtual std::string name() const = 0;

    /// 检测：输出 → 候选框
    virtual BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const {
        (void)out; (void)ctx; return {};
    }

    /// 分割：输出 → 带 mask 系数的候选框
    virtual std::vector<Candidate<std::array<float, 32>>> decode_segment(
        const TensorSet& out, const DecodeContext& ctx) const {
        (void)out; (void)ctx; return {};
    }

    /// 姿态：输出 → 带关键点的候选框
    virtual std::vector<Candidate<std::vector<Keypoint>>> decode_pose(
        const TensorSet& out, const DecodeContext& ctx) const {
        (void)out; (void)ctx; return {};
    }

    /// 旋转框：输出 → OBB 候选
    virtual OBBBoxArray decode_obb(const TensorSet& out, const DecodeContext& ctx) const {
        (void)out; (void)ctx; return {};
    }
};

// ============================================================
// 具体格式解码器（每个模型导出格式一个类）
// ============================================================

/// YOLOv5 — anchor-based
class V5Decoder : public Decoder {
public:
    std::string name() const override { return "v5"; }
    BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const override;
};

/// YOLOX — anchor-free
class YOLOXDecoder : public Decoder {
public:
    std::string name() const override { return "yolox"; }
    BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const override;
};

/// YOLOv8 / v11 / v26 — anchor-free（DFL），三者输出格式一致
/// 同时支持 detect / segment / pose / obb 四种任务头（v11+ 起各任务共享主干格式）
class V8Decoder : public Decoder {
public:
    std::string name() const override { return "v8"; }

    BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const override;

    std::vector<Candidate<std::array<float, 32>>> decode_segment(
        const TensorSet& out, const DecodeContext& ctx) const override;

    std::vector<Candidate<std::vector<Keypoint>>> decode_pose(
        const TensorSet& out, const DecodeContext& ctx) const override;

    OBBBoxArray decode_obb(const TensorSet& out, const DecodeContext& ctx) const override;
};

/// PPYOLOE — anchor-free，cls 走 sigmoid 无独立 obj
class PPYOLOEDecoder : public Decoder {
public:
    std::string name() const override { return "ppyoloe"; }
    BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const override;
};

/// 按模型类型创建解码器
/// @param type 模型类型（v5 / yolox / v8 系 / ppyoloe）
std::shared_ptr<Decoder> create_decoder(ModelType type);

} // namespace yolo_onnx