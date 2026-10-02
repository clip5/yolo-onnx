#include "process/postprocess/decoder/decoder.hpp"

#include <cmath>
#include <iostream>

namespace yolo_onnx {

namespace {

/// 判断 YOLOX 输出的前 4 列是「已解码的像素坐标」还是「原始 logit」。
///
/// 带 decoder 的导出（官方 export_demo / decode_spatial 里把解码拼进图）直接
/// 给出像素 xywh；不带 decoder 的导出给的是 reg 分支原始 logit。
///
/// 判别依据（不能用「有没有负值」——框中心允许略微越出图像边界，实测出现过
/// col1 = -1.06，若要求全非负就会把已解码的输出误判成 logit，进而用网格反推
/// 算出 27x27 这类量化框）：
///   * 量级：像素坐标的 max 与输入尺寸同阶（几百），logit 则集中在 0 附近（几）；
///   * 符号：logit 大致一半为负，像素坐标几乎全非负（少数越界的小负值可容忍）。
bool looks_decoded(const float* data, int num_rows, int channels, int input_size) {
    if (data == nullptr || num_rows <= 0) return false;

    float vmax = 0.0f;
    int negative = 0;
    long long seen = 0;

    // 只抽样前若干行即可判定，省去全量扫描的开销
    const int step = num_rows > 512 ? num_rows / 512 : 1;
    for (int i = 0; i < num_rows; i += step) {
        const float* row = data + (size_t)i * channels;
        for (int c = 0; c < 4; c++) {
            const float v = row[c];
            if (v > vmax) vmax = v;
            if (v < 0.0f) negative++;
            seen++;
        }
    }
    if (seen == 0) return false;

    // 阈值取「明显大于 logit 量级、又不依赖具体输入尺寸」的折中值
    const float mag_threshold = std::max(20.0f, 0.05f * (float)input_size);
    const float neg_ratio = (float)negative / (float)seen;
    return vmax > mag_threshold && neg_ratio < 0.01f;
}

} // namespace

// ============================================================
// YOLOX — anchor-free
// ============================================================
// 导出格式 A: 单个输出 [1, num_grids, 5+C]（box-major，按 stride 8/16/32 顺序拼接）
// 导出格式 B: 3 个输出 [1, 5+C, H, W]
//
// 两种 box 布局（由 looks_decoded() 在运行时判别，不依赖 metadata）：
//   (a) 带 decoder：cols 0:4 已是像素 xywh，直接使用
//   (b) 不带 decoder：cols 0:4 是原始 logit，用官方解码
//         center = (sigmoid(t)*2 - 0.5 + grid) * stride
//         size   = (sigmoid(t)*2)^2 * stride
//
// 注意：第 5 列起（obj + 各类分数）在两种导出里都**已经过 sigmoid**
// （Concat 的 cls 分支带 Sigmoid），值域恒为 [0,1]。这里不能再补一次
// sigmoid——那会把所有低分项都抬到 0.5 以上，导致几乎所有格子都越过阈值
// （历史实现正是如此，8400 个格子全部通过）。
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
        const bool decoded = looks_decoded(data, num_grids, channels,
                                           std::max(ctx.input_width, ctx.input_height));

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

                const float obj_conf = row[4];
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    const float cls = row[5 + c];
                    if (cls > max_cls) { max_cls = cls; max_cls_id = c; }
                }

                const float score = obj_conf * max_cls;
                if (score < score_thresh) continue;

                float bx, by, bw, bh;
                if (decoded) {
                    // 带 decoder 的导出：前 4 列已是像素 xywh，直接用
                    bx = row[0]; by = row[1]; bw = row[2]; bh = row[3];
                } else {
                    // 不带 decoder：原始 logit，按官方公式反推网格坐标
                    bx = (sigmoid(row[0]) * 2.0f - 0.5f + gi) * stride;
                    by = (sigmoid(row[1]) * 2.0f - 0.5f + gj) * stride;
                    bw = std::pow(sigmoid(row[2]) * 2.0f, 2) * stride;
                    bh = std::pow(sigmoid(row[3]) * 2.0f, 2) * stride;
                }

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
                const float obj = data[4 + h * width + w];
                float max_cls = 0.0f;
                int max_cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    const float cls = data[5 + c + h * width + w];
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