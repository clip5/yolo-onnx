#include "process/postprocess/decoder/decoder.hpp"

#include <cmath>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// YOLOv8 / YOLOv11 / YOLO26 — anchor-free，DFL
// ============================================================
// 三者输出格式一致，共用本解码器。
//
// 导出格式 A: 单个输出 [1, C, num_boxes]，按 stride 8/16/32 顺序拼接
//   C = 4+num_classes          → 已是解码后的 cx,cy,w,h（像素）
//   C = 4*reg_max + num_classes → DFL 分布分支，box 由4 边各自 softmax 期望得到
// 导出格式 B: 3 个输出 [1, C, H, W]（每层一个），值为原始 logit
//
// 注：类分数的 sigmoid 探测（raw∈[0,1] 视为已激活）沿用历史实现——
// 部分导出已带 sigmoid、部分没有，靠取值范围区分。这一启发式本身有
// 已知问题（会产生 0 宽高的退化框），本次迁移保持行为不变，待单独排查。
BoxArray V8Decoder::decode_detect(const TensorSet& out, const DecodeContext& ctx) const {
    BoxArray boxes;
    if (out.empty()) return boxes;

    const int num_classes = ctx.num_classes;
    const float score_thresh = ctx.score_thresh;

    // ---- 格式 A: 单个拼接输出 [1, C, num_boxes] ----
    if (out.shape(0).size() == 3) {
        const float* data = out.data(0);
        const int channels = (int)out.shape(0)[1];
        const int num_boxes = (int)out.shape(0)[2];
        if (channels <= 0 || num_boxes <= 0) return boxes;

        // DFL 判定：通道数远超 4+C 时，视为带 reg_max*4 分布分支
        const bool has_dfl = channels > 4 + num_classes + 32;
        const int reg_max = has_dfl ? (channels - num_classes) / 4 : 0;
        const int cls_start = has_dfl ? 4 * reg_max : 4;

        int grid_offset = 0;
        for (int level = 0; level < 3; level++) {
            const int stride = ctx.stride(level);
            const int grid_w = ctx.grid_w(level);
            const int count  = ctx.grid_count(level);

            for (int g = 0; g < count; g++) {
                const int idx = grid_offset + g;
                if (idx >= num_boxes) break;   // 网格由实际输入推导，夹在张量范围内
                const int gi = g % grid_w;
                const int gj = g / grid_w;

                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    const float raw  = data[(size_t)(cls_start + c) * num_boxes + idx];
                    const float cls = (raw >= 0.0f && raw <= 1.0f) ? raw : sigmoid(raw);
                    if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                }
                if (max_cls < score_thresh) continue;

                float cx, cy, bw, bh;
                if (has_dfl) {
                    // DFL：每条边对 reg_max 个logit 做 softmax，期望值即该边偏移。
                    // 两遍扫描（先sum_exp 再累加期望），不缓存 probs[]，避免
                    // reg_max 超过固定数组长度时的越界。
                    float dfl_val[4] = {0, 0, 0, 0};
                    for (int s = 0; s < 4; s++) {
                        const float* logits = data + (size_t)(s * reg_max) * num_boxes + idx;
                        float sum_exp = 0.0f;
                        for (int i = 0; i < reg_max; i++) sum_exp += std::exp(logits[i]);
                        if (sum_exp > 0.0f) {
                            for (int i = 0; i < reg_max; i++) {
                                dfl_val[s] += (std::exp(logits[i]) / sum_exp) * i;
                            }
                        }
                    }
                    cx = (gi + 0.5f) * stride;
                    cy = (gj + 0.5f) * stride;
                    bw = (dfl_val[0] + dfl_val[2]) * stride;
                    bh = (dfl_val[1] + dfl_val[3]) * stride;
                } else {
                    cx = data[(size_t)0 * num_boxes + idx];
                    cy = data[(size_t)1 * num_boxes + idx];
                    bw = data[(size_t)2 * num_boxes + idx];
                    bh = data[(size_t)3 * num_boxes + idx];
                }

                boxes.emplace_back(cx - bw / 2.0f, cy - bh / 2.0f,
                                   cx + bw / 2.0f, cy + bh / 2.0f, max_cls, max_cls_id);
            }
            grid_offset += count;
        }
        return boxes;
    }

    // ---- 格式 B: 3 个分层输出 [1, C, H, W] ----
    if (out.size() < 3 || out.shape(0).size() != 4) return boxes;

    for (int level = 0; level < 3; level++) {
        const auto& shape = out.shape(level);
        if (shape.size() != 4) continue;

        const float* data = out.data(level);
        const int height = (int)shape[2];
        const int width  = (int)shape[3];
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

                const float cx = data[(size_t)0 + h * width + w];
                const float cy = data[(size_t)1 + h * width + w];
                const float bw = data[(size_t)2 + h * width + w];
                const float bh = data[(size_t)3 + h * width + w];

                const float bx      = (sigmoid(cx) + w) * stride;
                const float by      = (sigmoid(cy) + h) * stride;
                const float box_w   = bw * stride;
                const float box_h   = bh * stride;

                boxes.emplace_back(bx - box_w / 2.0f, by - box_h / 2.0f,
                                   bx + box_w / 2.0f, by + box_h / 2.0f, max_cls, max_cls_id);
            }
        }
    }
    return boxes;
}

} // namespace yolo_onnx