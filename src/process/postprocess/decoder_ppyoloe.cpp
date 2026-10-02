#include "process/postprocess/decoder.hpp"

#include <cmath>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// PPYOLOE — anchor-free，cls 走 sigmoid，无独立 obj 置信度
// ============================================================
// 导出格式 A: 6 个输出 [cls0,cls1,cls2, reg0,reg1,reg2]
//             cls_i 为 [1,C,H,W]，reg_i 为 [1,4,H,W]，同层成对对应 stride 8/16/32
// 导出格式 B: 3 个输出 [1, 4+C, H, W]（cls 与 reg 合并在同一张量）
// 导出格式 C: 1 个输出 [1, 4+C, num_boxes]（拼接，按 stride 顺序）
// 解码：center = (sigmoid(d) + grid) * stride，size = exp(d) * stride
BoxArray PPYOLOEDecoder::decode_detect(const TensorSet& out, const DecodeContext& ctx) const {
    BoxArray boxes;
    const int num_classes = ctx.num_classes;
    const float score_thresh = ctx.score_thresh;

    // ---- 格式 A: 6 个输出（3 cls + 3 reg） ----
    if (out.size() == 6) {
        for (int level = 0; level < 3; level++) {
            const auto& cls_shape = out.shape(level);
            const auto& reg_shape = out.shape(level + 3);
            if (cls_shape.size() != 4 || reg_shape.size() != 4) continue;

            const float* cls_data = out.data(level);
            const float* reg_data = out.data(level + 3);
            const int cls_h = (int)cls_shape[2], cls_w = (int)cls_shape[3];
            const int reg_h = (int)reg_shape[2], reg_w = (int)reg_shape[3];
            const int stride = ctx.stride(level);

            for (int h = 0; h < cls_h; h++) {
                for (int w = 0; w < cls_w; w++) {
                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        const float cls = sigmoid(cls_data[(size_t)c * cls_h * cls_w + h * cls_w + w]);
                        if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                    }
                    if (max_cls < score_thresh) continue;

                    const int off = h * reg_w + w;
                    const float dx = reg_data[(size_t)0 * reg_h * reg_w + off];
                    const float dy = reg_data[(size_t)1 * reg_h * reg_w + off];
                    const float dw = reg_data[(size_t)2 * reg_h * reg_w + off];
                    const float dh = reg_data[(size_t)3 * reg_h * reg_w + off];

                    const float cx = (sigmoid(dx) + w) * stride;
                    const float cy = (sigmoid(dy) + h) * stride;
                    const float bw = std::exp(dw) * stride;
                    const float bh = std::exp(dh) * stride;

                    boxes.emplace_back(cx - bw / 2.0f, cy - bh / 2.0f,
                                       cx + bw / 2.0f, cy + bh / 2.0f, max_cls, max_cls_id);
                }
            }
        }
        return boxes;
    }

    // ---- 格式 B: 3 个输出 [1, 4+C, H, W] ----
    if (out.size() == 3) {
        for (int level = 0; level < 3; level++) {
            const auto& shape = out.shape(level);
            if (shape.size() != 4) continue;
            if ((int)shape[1] < 4 + num_classes) continue;

            const float* data = out.data(level);
            const int height = (int)shape[2], width = (int)shape[3];
            const int stride = ctx.stride(level);

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        const float cls = sigmoid(data[(4 + c) + (size_t)h * width + w]);
                        if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                    }
                    if (max_cls < score_thresh) continue;

                    const int off = h * width + w;
                    const float dx = data[(size_t)0 + off];
                    const float dy = data[(size_t)1 + off];
                    const float dw = data[(size_t)2 + off];
                    const float dh = data[(size_t)3 + off];

                    const float cx = (sigmoid(dx) + w) * stride;
                    const float cy = (sigmoid(dy) + h) * stride;
                    const float bw = std::exp(dw) * stride;
                    const float bh = std::exp(dh) * stride;

                    boxes.emplace_back(cx - bw / 2.0f, cy - bh / 2.0f,
                                       cx + bw / 2.0f, cy + bh / 2.0f, max_cls, max_cls_id);
                }
            }
        }
        return boxes;
    }

    // ---- 格式 C: 1 个拼接输出 [1, 4+C, num_boxes] ----
    if (out.size() == 1) {
        const auto& shape = out.shape(0);
        if (shape.size() != 3 && shape.size() != 2) {
            std::cerr << "[PPYOLOEDecoder] Unexpected single output shape" << std::endl;
            return boxes;
        }
        const float* data = out.data(0);
        const int num_boxes = (int)shape.back();
        if (num_boxes <= 0) return boxes;

        int grid_offset = 0;
        for (int level = 0; level < 3; level++) {
            const int stride = ctx.stride(level);
            const int grid_w = ctx.grid_w(level);
            const int count  = ctx.grid_count(level);

            for (int g = 0; g < count; g++) {
                const int idx = grid_offset + g;
                if (idx >= num_boxes) break;

                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    const float cls = sigmoid(data[(size_t)(4 + c) * num_boxes + idx]);
                    if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                }
                if (max_cls < score_thresh) continue;

                const int gi = idx % grid_w;
                const int gj = idx / grid_w;
                const float dx = data[(size_t)0 * num_boxes + idx];
                const float dy = data[(size_t)1 * num_boxes + idx];
                const float dw = data[(size_t)2 * num_boxes + idx];
                const float dh = data[(size_t)3 * num_boxes + idx];

                const float cx = (sigmoid(dx) + gi) * stride;
                const float cy = (sigmoid(dy) + gj) * stride;
                const float bw = std::exp(dw) * stride;
                const float bh = std::exp(dh) * stride;

                boxes.emplace_back(cx - bw / 2.0f, cy - bh / 2.0f,
                                   cx + bw / 2.0f, cy + bh / 2.0f, max_cls, max_cls_id);
            }
            grid_offset += count;
        }
    }
    return boxes;
}

} // namespace yolo_onnx