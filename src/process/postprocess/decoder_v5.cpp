#include "process/postprocess/decoder.hpp"

#include <cmath>
#include <iostream>

namespace yolo_onnx {

// ============================================================
// YOLOv5 — anchor-based
// ============================================================
// 导出格式 A: 单个输出 [1, num_boxes, 5+C]
//   box = cx,cy,w,h（已是像素坐标），score = obj * max_cls，obj/cls 已 sigmoid
// 导出格式 B: 3 个输出 [1, 3*(5+C), H, W]（stride 8/16/32），原始值需解码
//
// anchors_[level][anchor] 为该层 anchor 的宽高（像素）
static constexpr int kV5Anchors[3][3][2] = {
    {{10, 13}, {16, 30}, {33, 23}},     // P3 / stride 8
    {{30, 61}, {62, 45}, {59, 119}},    // P4 / stride 16
    {{116, 90}, {156, 198}, {373, 326}} // P5 / stride 32
};
static constexpr int kV5NumAnchors = 3;

BoxArray V5Decoder::decode_detect(const TensorSet& out, const DecodeContext& ctx) const {
    BoxArray boxes;
    const int num_classes = ctx.num_classes;
    const float score_thresh = ctx.score_thresh;
    const int bbox_ch = 5 + num_classes;

    // ---- 格式 A: 单个拼接输出 [1, num_boxes, 5+C] ----
    if (out.size() == 1) {
        const auto& shape = out.shape(0);
        if (shape.size() != 3 || (int)shape[2] != bbox_ch) {
            std::cerr << "[V5Decoder] Unexpected single output shape" << std::endl;
            return boxes;
        }
        const float* data = out.data(0);
        const int num_boxes = (int)shape[1];

        for (int i = 0; i < num_boxes; i++) {
            const float* row = data + (size_t)i * bbox_ch;
            const float obj = row[4];
            float max_cls = 0.0f;
            int max_cls_id = -1;
            for (int c = 0; c < num_classes; c++) {
                if (row[5 + c] > max_cls) { max_cls = row[5 + c]; max_cls_id = c; }
            }
            const float score = obj * max_cls;
            if (max_cls_id < 0 || score < score_thresh) continue;

            const float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
            boxes.emplace_back(cx - bw / 2.0f, cy - bh / 2.0f,
                               cx + bw / 2.0f, cy + bh / 2.0f, score, max_cls_id);
        }
        return boxes;
    }

    // ---- 格式 B: 3 个分层输出 [1, 3*(5+C), H, W] ----
    if (out.size() != 3) {
        std::cerr << "[V5Decoder] Expected 1 or 3 outputs, got "
                  << out.size() << std::endl;
        return boxes;
    }

    for (int level = 0; level < 3; level++) {
        const auto& shape = out.shape(level);
        if (shape.size() != 4) continue;

        const float* data = out.data(level);
        const int channels = (int)shape[1];
        const int height   = (int)shape[2];
        const int width    = (int)shape[3];
        const int stride   = ctx.stride(level);

        if (channels != kV5NumAnchors * bbox_ch) continue;

        for (int a = 0; a < kV5NumAnchors; a++) {
            const int anchor_offset = a * bbox_ch;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    const float obj = sigmoid(data[anchor_offset + 4 + h * width + w]);

                    float max_cls = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        const float cls = sigmoid(data[anchor_offset + 5 + c + h * width + w]);
                        if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                    }

                    const float score = obj * max_cls;
                    if (score < score_thresh) continue;

                    const int idx_off = anchor_offset + h * width + w;
                    const float tx = data[idx_off];
                    const float ty = data[anchor_offset + 1 + h * width + w];
                    const float tw = data[anchor_offset + 2 + h * width + w];
                    const float th = data[anchor_offset + 3 + h * width + w];

                    const float bx = (sigmoid(tx) + w) * stride;
                    const float by = (sigmoid(ty) + h) * stride;
                    const float bw = std::exp(tw) * kV5Anchors[level][a][0];
                    const float bh = std::exp(th) * kV5Anchors[level][a][1];

                    boxes.emplace_back(bx - bw / 2.0f, by - bh / 2.0f,
                                       bx + bw / 2.0f, by + bh / 2.0f, score, max_cls_id);
                }
            }
        }
    }
    return boxes;
}

} // namespace yolo_onnx