#include "process/postprocess/postprocess.hpp"
#include "process/postprocess/postprocess_core.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace yolo_onnx {

// ============================================================
// PostProcessSegment — 实例分割
// ============================================================
// 流程：decode（框+mask 系数）→ 按框过滤/还原/NMS → 由 proto mask 合成实例掩码
//
// 模型输出两个张量：
//   1. 检测 + mask 系数：[1, 4+num_classes+32, num_boxes]
//   2. proto masks       ：[1, 32, H, W]（H,W 通常为输入的 1/4）
// mask 系数与候选框按下标绑定，因此先按框选出保留项，再回取对应系数。

namespace {
constexpr int kNumMasks = 32;   // mask 系数通道数
}

int PostProcessSegment::find_proto_index(const TensorSet& outputs) {
    for (size_t i = 0; i < outputs.size(); i++) {
        const auto& shape = outputs.shape(i);
        // proto mask: [1, 32, H, W]，H/W 通常为输入的 1/4
        if (shape.size() == 4 && shape[1] == kNumMasks && shape[2] > 0 && shape[3] > 0) {
            return (int)i;
        }
    }
    return -1;
}

std::vector<Mask> PostProcessSegment::compose_masks(
    const TensorSet& outputs, int proto_idx,
    const std::vector<std::array<float, kNumMasks>>& coeffs,
    const LetterboxInfo& lb,
    const std::vector<Box>& boxes,
    int input_width) {

    std::vector<Mask> masks;
    const auto& proto_shape = outputs.shape(proto_idx);
    const int proto_h = (int)proto_shape[2];
    const int proto_w = (int)proto_shape[3];
    const int orig_w  = lb.orig_w;
    const int orig_h  = lb.orig_h;
    if (proto_h <= 0 || proto_w <= 0) return masks;

    const float* proto = outputs.data(proto_idx);

    for (size_t k = 0; k < boxes.size() && k < coeffs.size(); k++) {
        const auto& coeff = coeffs[k];
        const Box& box    = boxes[k];

        // mask = sigmoid(proto @ coeffs)，proto 为 [32, proto_h, proto_w]
        Mask mask(proto_w, proto_h);
        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                float val = 0.0f;
                for (int c = 0; c < kNumMasks; c++) {
                    val += proto[(size_t)c * proto_h * proto_w + (size_t)h * proto_w + w] * coeff[c];
                }
                mask.data[(size_t)h * proto_w + w] = sigmoid(val);
            }
        }

        // 裁剪到框区域（proto mask 空间）
        // 注：沿用历史实现的坐标换算（对已还原到原图的 box 再减一次 padding），
        // 该换算本身存疑，见 AGENTS.md Gotchas，本次保持行为不变。
        const float proto_scale = (float)proto_w / std::max(1, input_width);
        int x1 = std::max(0, (int)((box.x1 - lb.pad_left) * proto_scale));
        int y1 = std::max(0, (int)((box.y1 - lb.pad_top)  * proto_scale));
        int x2 = std::min(proto_w - 1, (int)((box.x2 - lb.pad_left) * proto_scale));
        int y2 = std::min(proto_h - 1, (int)((box.y2 - lb.pad_top)  * proto_scale));

        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                if (w < x1 || w > x2 || h < y1 || h > y2) {
                    mask.data[(size_t)h * proto_w + w] = 0.0f;
                }
            }
        }

        // 缩放到原图尺寸并二值化
        cv::Mat mask_cv(proto_h, proto_w, CV_32FC1, mask.data.data());
        cv::Mat resized, binary;
        cv::resize(mask_cv, resized, cv::Size(orig_w, orig_h));
        cv::threshold(resized, binary, 0.5f, 1.0f, cv::THRESH_BINARY);

        Mask final_mask(orig_w, orig_h);
        for (int h = 0; h < orig_h; h++) {
            for (int w = 0; w < orig_w; w++) {
                final_mask.data[(size_t)h * orig_w + w] = binary.at<float>(h, w);
            }
        }
        masks.push_back(std::move(final_mask));
    }
    return masks;
}

InferResult PostProcessSegment::forward(const TensorSet& outputs,
                                        const LetterboxInfo& lb) const {
    auto cands = decoder_->decode_segment(outputs, ctx_);
    if (cands.empty()) return SegmentResult{};

    std::vector<Box> boxes;
    boxes.reserve(cands.size());
    for (const auto& c : cands) boxes.push_back(c.box);

    auto keep = select_indices(boxes, lb, params_);
    if (keep.empty()) return SegmentResult{};

    SegmentResult result;
    result.boxes.reserve(keep.size());
    std::vector<std::array<float, kNumMasks>> kept_coeffs;
    kept_coeffs.reserve(keep.size());

    for (int i : keep) {
        Box b = cands[i].box;
        scale_box(b, lb);                       // 还原到原图坐标
        result.boxes.push_back(b);
        kept_coeffs.push_back(cands[i].extra);  // 系数与框同下标，一并保留
    }

    const int proto_idx = find_proto_index(outputs);
    if (proto_idx >= 0) {
        result.masks = compose_masks(outputs, proto_idx, kept_coeffs, lb, result.boxes,
                                     ctx_.input_width);
    }
    return result;
}

} // namespace yolo_onnx