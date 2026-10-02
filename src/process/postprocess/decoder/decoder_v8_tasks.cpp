#include "process/postprocess/decoder/decoder.hpp"

#include <cmath>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// V8Decoder — segment / pose / obb 任务头
// ============================================================
// 这三种任务与 detect 共享同一种主干输出格式，区别只在通道尾部附加的信息：
//   segment: [cx,cy,w,h, cls0..clsC, mask_coeff×32]
//   pose:    [cx,cy,w,h, cls0..clsC, (x,y,vis)×num_kpts]
//   obb:     [cx,cy,w,h, angle, cls0..clsC]
//
// 支持两种导出布局（沿用历史行为）：
//   3D [1, C, N]      —— 三个 stride 层的格子按 8/16/32 顺序拼接
//   4D [1, C, H, W]   —— 单层网格（训练中间输出）
//
// 网格尺寸一律由 DecodeContext 的实际输入尺寸推导，并做越界保护——
// 旧实现在此处硬编码 80/40/20 与 6400/1600/400，对非正方形输入会读越界。

namespace {

/// 遍历输出中的所有网格格子，兼容 3D/4D 两种布局。
/// 回调签名: fn(channel_stride, cell_idx, gi, gj, stride)
///   channel_stride —— 该布局下「每通道平面」的元素数（3D 为 N，4D 为 H*W）
///   cell_idx       —— 格子在平面内的偏移（3D 为全局 idx，4D 为 h*W+w）
///   gi, gj         —— 格子的列/行号（用于网格坐标反推）
///   stride         —— 该格子的步长
template <typename Fn>
bool walk_cells(const TensorSet& out, const DecodeContext& ctx, Fn&& fn) {
    const auto& shape = out.shape(0);
    if (shape.size() < 3) return false;

    // ---- 3D [1, C, N]：三level 拼接 ----
    if (shape.size() == 3) {
        const int num_boxes = (int)shape[2];
        if (num_boxes <= 0) return false;
        const int channel_stride = num_boxes;

        int grid_offset = 0;
        for (int level = 0; level < 3; level++) {
            const int stride = ctx.stride(level);
            const int grid_w = ctx.grid_w(level);
            const int count  = ctx.grid_count(level);

            for (int g = 0; g < count; g++) {
                const int idx = grid_offset + g;
                if (idx >= num_boxes) return true;   // 超出张量范围，提前结束
                fn(channel_stride, idx, g % grid_w, g / grid_w, stride);
            }
            grid_offset += count;
        }
        return true;
    }

    // ---- 4D [1, C, H, W]：单层网格 ----
    if (shape.size() == 4) {
        const int height = (int)shape[2];
        const int width  = (int)shape[3];
        if (height <= 0 || width <= 0) return false;
        const int channel_stride = height * width;
        const int stride = (width > 0) ? ctx.input_width / width : 1;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                fn(channel_stride, h * width + w, w, h, stride);
            }
        }
        return true;
    }

    return false;
}

/// 从张量读通道值（兼容两种布局：cs 为通道平面大小，idx 为格内偏移）
inline float channel_at(const float* data, int cs, int ch, int idx) {
    return data[(size_t)ch * cs + idx];
}

/// 在「原始 logit ∈ [0,1]」与「未激活 logit」之间自动判断是否需要补sigmoid。
/// 沿用历史启发式：部分导出已带 sigmoid、部分没有。已知该启发式有缺陷
/// （会产生 0 宽高的退化框），本次迁移保持行为不变，待单独排查。
inline float maybe_sigmoid(float raw) {
    return (raw >= 0.0f && raw <= 1.0f) ? raw : sigmoid(raw);
}

} // namespace

// ============================================================
// segment：解出 box + 32 个 mask 系数
// ============================================================
std::vector<V8Decoder::Candidate<std::array<float, 32>>>
V8Decoder::decode_segment(const TensorSet& out, const DecodeContext& ctx) const {
    std::vector<Candidate<std::array<float, 32>>> candidates;
    const float* data = out.data(0);
    if (!data) return candidates;

    const int num_classes  = ctx.num_classes;
    const float score_thresh = ctx.score_thresh;
    const int cls_start    = 4;
    const int coeff_start = 4 + num_classes;

    const auto& shape = out.shape(0);
    if (shape.size() < 3) return candidates;
    if ((int)shape[1] < coeff_start + 32) {
        std::cerr << "[V8Decoder:segment] channels " << (int)shape[1]
                  << " < required " << (coeff_start + 32) << std::endl;
        return candidates;
    }

    walk_cells(out, ctx, [&](int cs, int idx, int gi, int gj, int stride) {
        float max_cls = 0.0f;
        int max_cls_id = -1;
        for (int c = 0; c < num_classes; c++) {
            const float cls = maybe_sigmoid(channel_at(data, cs, cls_start + c, idx));
            if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
        }
        if (max_cls < score_thresh) return;

        const float cx = channel_at(data, cs, 0, idx);
        const float cy = channel_at(data, cs, 1, idx);
        const float bw = channel_at(data, cs, 2, idx);
        const float bh = channel_at(data, cs, 3, idx);

        Candidate<std::array<float, 32>> cand;
        // 3D 为已解码的像素坐标；4D 为网格空间 logit，需 sigmoid + grid 偏移
        float bx = cx, by = cy, box_w = bw, box_h = bh;
        if (out.shape(0).size() == 4) {
            bx = (sigmoid(cx) + gi) * stride;
            by = (sigmoid(cy) + gj) * stride;
            box_w = bw * stride;
            box_h = bh * stride;
        }
        cand.box = Box(bx - box_w / 2.0f, by - box_h / 2.0f,
                       bx + box_w / 2.0f, by + box_h / 2.0f, max_cls, max_cls_id);
        for (int m = 0; m < 32; m++) {
            cand.extra[m] = channel_at(data, cs, coeff_start + m, idx);
        }
        candidates.push_back(std::move(cand));
    });

    return candidates;
}

// ============================================================
// pose：解出box + num_keypoints 个关键点
// ============================================================
std::vector<V8Decoder::Candidate<std::vector<Keypoint>>>
V8Decoder::decode_pose(const TensorSet& out, const DecodeContext& ctx) const {
    std::vector<Candidate<std::vector<Keypoint>>> candidates;
    const float* data = out.data(0);
    if (!data) return candidates;

    const int num_classes   = ctx.num_classes;
    const int num_kpts      = ctx.num_keypoints;
    const float score_thresh = ctx.score_thresh;
    const int kpt_start     = 4 + num_classes;
    constexpr int kpt_dim    = 3;   // x, y, visibility

    const auto& shape = out.shape(0);
    if (shape.size() < 3) return candidates;
    if ((int)shape[1] < kpt_start + num_kpts * kpt_dim) {
        std::cerr << "[V8Decoder:pose] channels " << (int)shape[1]
                  << " < required " << (kpt_start + num_kpts * kpt_dim) << std::endl;
        return candidates;
    }

    const bool is_chw = (shape.size() == 4);

    walk_cells(out, ctx, [&](int cs, int idx, int gi, int gj, int stride) {
        float max_cls = 0.0f;
        int max_cls_id = -1;
        for (int c = 0; c < num_classes; c++) {
            const float cls = sigmoid(channel_at(data, cs, 4 + c, idx));
            if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
        }
        if (max_cls < score_thresh) return;

        const float cx = channel_at(data, cs, 0, idx);
        const float cy = channel_at(data, cs, 1, idx);
        const float bw = channel_at(data, cs, 2, idx);
        const float bh = channel_at(data, cs, 3, idx);

        Candidate<std::vector<Keypoint>> cand;
        float bx = cx, by = cy, box_w = bw, box_h = bh;
        if (is_chw) {
            bx      = (sigmoid(cx) + gi) * stride;
            by      = (sigmoid(cy) + gj) * stride;
            box_w   = bw * stride;
            box_h   = bh * stride;
        }
        cand.box = Box(bx - box_w / 2.0f, by - box_h / 2.0f,
                       bx + box_w / 2.0f, by + box_h / 2.0f, max_cls, max_cls_id);

        for (int k = 0; k < num_kpts; k++) {
            Keypoint kp;
            const float x = channel_at(data, cs, kpt_start + k * kpt_dim + 0, idx);
            const float y = channel_at(data, cs, kpt_start + k * kpt_dim + 1, idx);
            kp.visibility = channel_at(data, cs, kpt_start + k * kpt_dim + 2, idx);

            if (is_chw) {
                kp.x = (sigmoid(x) + gi) * stride;
                kp.y = (sigmoid(y) + gj) * stride;
            } else {
                // 3D 导出有两种：像素坐标，或 [0,1] 归一化 logit
                if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
                    kp.x = (sigmoid(x) + gi) * stride;
                    kp.y = (sigmoid(y) + gj) * stride;
                } else {
                    kp.x = x;
                    kp.y = y;
                }
            }
            cand.extra.push_back(kp);
        }
        candidates.push_back(std::move(cand));
    });

    return candidates;
}

// ============================================================
// obb：解出旋转框，通道位置 4 为角度
// ============================================================
OBBBoxArray V8Decoder::decode_obb(const TensorSet& out, const DecodeContext& ctx) const {
    OBBBoxArray candidates;
    const float* data = out.data(0);
    if (!data) return candidates;

    const int num_classes   = ctx.num_classes;
    const float score_thresh = ctx.score_thresh;
    constexpr int angle_pos = 4;
    const int cls_start     = 5;   // 角度固定在位置 4，类分数从5 开始

    const auto& shape = out.shape(0);
    if (shape.size() < 3) return candidates;
    if ((int)shape[1] < cls_start + num_classes) {
        std::cerr << "[V8Decoder:obb] channels " << (int)shape[1]
                  << " < required " << (cls_start + num_classes) << std::endl;
        return candidates;
    }

    const bool is_chw = (shape.size() == 4);

    walk_cells(out, ctx, [&](int cs, int idx, int gi, int gj, int stride) {
        float max_cls = 0.0f;
        int max_cls_id = -1;
        for (int c = 0; c < num_classes; c++) {
            const float cls = sigmoid(channel_at(data, cs, cls_start + c, idx));
            if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
        }
        if (max_cls < score_thresh) return;

        const float cx = channel_at(data, cs, 0, idx);
        const float cy = channel_at(data, cs, 1, idx);
        const float bw = channel_at(data, cs, 2, idx);
        const float bh = channel_at(data, cs, 3, idx);
        const float angle = sigmoid(channel_at(data, cs, angle_pos, idx)) * CV_PI;

        float bx, by, box_w, box_h;
        if (is_chw) {
            bx    = (sigmoid(cx) + gi) * stride;
            by    = (sigmoid(cy) + gj) * stride;
            box_w = bw * stride;
            box_h = bh * stride;
        } else if (cx >= 0.0f && cx <= 1.0f && cy >= 0.0f && cy <= 1.0f) {
            // 3D 导出为 [0,1] 归一化 logit
            bx    = (sigmoid(cx) + gi) * stride;
            by    = (sigmoid(cy) + gj) * stride;
            box_w = bw * stride;
            box_h = bh * stride;
        } else {
            // 3D 导出已是像素坐标
            bx = cx; by = cy; box_w = bw; box_h = bh;
        }

        candidates.emplace_back(bx, by, box_w, box_h, angle, max_cls, max_cls_id);
    });

    return candidates;
}

} // namespace yolo_onnx