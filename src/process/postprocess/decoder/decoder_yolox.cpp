#include "process/postprocess/decoder/decoder.hpp"

#include <cmath>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// YOLOX — anchor-free
// ============================================================
// 导出格式 A: 单个输出 [1, num_grids, 5+C]（box-major，按 stride 8/16/32 顺序拼接）
// 导出格式 B: 3 个输出 [1, 5+C, H, W]
// 官方解码公式：center = (sigmoid(t)*2 - 0.5 + grid) * stride
//                size   = (sigmoid(t)*2)^2 * stride
BoxArray YOLOXDecoder::decode_detect(const TensorSet& out, const DecodeContext& ctx) const {
    BoxArray boxes;
    if (out.empty()) return boxes;

    const int num_classes = ctx.num_classes;
    const float score_thresh = ctx.score_thresh;
    const int bbox_ch = 5 + num_classes;

    // ---- 格式 A: 单个输出 [1, num_grids, 5+C] ----
    if (out.size() == 1) {
        const auto& shape = out.shape(0);
        if (shape.size() != 3) {
            std::cerr << "[YOLOXDecoder] Unexpected single output shape: "
                      << shape.size() << "D" << std::endl;
            return boxes;
        }
        const float* data = out.data(0);
        const int channels = (int)shape[2];
        if (channels != bbox_ch) {
            std::cerr << "[YOLOXDecoder] Unexpected channel count: "
                      << channels << std::endl;
            return boxes;
        }
        const int num_grids = (int)shape[1];

        int grid_offset = 0;
        for (int level = 0; level < 3; level++) {
            const int stride = ctx.stride(level);
            const int grid_w = ctx.grid_w(level);
            const int count  = ctx.grid_count(level);

            for (int g = 0; g < count; g++) {
                const int idx = grid_offset + g;
                // 网格按实际输入尺寸推导，并夹在张量范围内，避免读越界
                if (idx >= num_grids) break;

                const int gi = g % grid_w;
                const int gj = g / grid_w;
                const float* row = data + (size_t)idx * channels;

                const float obj_conf = sigmoid(row[4]);
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    const float cls = sigmoid(row[5 + c]);
                    if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                }

                const float score = obj_conf * max_cls;
                if (score < score_thresh) continue;

                const float bx = (sigmoid(row[0]) * 2.0f - 0.5f + gi) * stride;
                const float by = (sigmoid(row[1]) * 2.0f - 0.5f + gj) * stride;
                const float bw = std::pow(sigmoid(row[2]) * 2.0f, 2) * stride;
                const float bh = std::pow(sigmoid(row[3]) * 2.0f, 2) * stride;

                boxes.emplace_back(bx - bw / 2.0f, by - bh / 2.0f,
                                   bx + bw / 2.0f, by + bh / 2.0f, score, max_cls_id);
            }
            grid_offset += count;
        }
        return boxes;
    }

    // ---- 格式 B: 3 个输出 [1, 5+C, H, W] ----
    if (out.size() < 3) return boxes;

    for (int level = 0; level < 3; level++) {
        const auto& shape = out.shape(level);
        if (shape.size() != 4) continue;

        const float* data = out.data(level);
        const int channels = (int)shape[1];
        const int height   = (int)shape[2];
        const int width    = (int)shape[3];
        const int stride   = ctx.stride(level);
        if (channels != bbox_ch) continue;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                const float obj = sigmoid(data[4 + h * width + w]);
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    const float cls = sigmoid(data[5 + c + h * width + w]);
                    if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                }
                const float score = obj * max_cls;
                if (score < score_thresh) continue;

                const float bx = (sigmoid(data[0 + h * width + w]) * 2.0f - 0.5f + w) * stride;
                const float by = (sigmoid(data[1 + h * width + w]) * 2.0f - 0.5f + h) * stride;
                const float bw = std::pow(sigmoid(data[2 + h * width + w]) * 2.0f, 2) * stride;
                const float bh = std::pow(sigmoid(data[3 + h * width + w]) * 2.0f, 2) * stride;

                boxes.emplace_back(bx - bw / 2.0f, by - bh / 2.0f,
                                   bx + bw / 2.0f, by + bh / 2.0f, score, max_cls_id);
            }
        }
    }
    return boxes;
}

} // namespace yolo_onnx