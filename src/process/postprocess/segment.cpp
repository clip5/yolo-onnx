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
    int input_width, int input_height) {

    std::vector<Mask> masks;
    const auto& proto_shape = outputs.shape(proto_idx);
    const int proto_h = (int)proto_shape[2];
    const int proto_w = (int)proto_shape[3];
    const int orig_w  = lb.orig_w;
    const int orig_h  = lb.orig_h;
    if (proto_h <= 0 || proto_w <= 0) return masks;
    if (input_width <= 0 || input_height <= 0) return masks;

    const float* proto = outputs.data(proto_idx);

    // ---- 坐标换算 --------------------------------------------------------
    // proto mask 的第 (h,w) 元素对应模型输入坐标：
    //     x_input = (w + 0.5) * input_width  / proto_w
    // 而传入的 box 已经被 scale_box() 还原到**原图**坐标，因此要先按 letterbox
    // 映射回模型输入空间，再按 proto 分辨率缩放：
    //     x_proto = (x_orig * scale + pad_left) * proto_w / input_width
    // 旧实现漏掉了「映射回模型输入空间」这一步（直接对原图坐标减 padding），
    // 导致裁剪窗口整体偏移，mask 与框对不上（实测 mask IoU ≈ 0.06）。
    const float sx = (float)proto_w / (float)input_width;
    const float sy = (float)proto_h / (float)input_height;

    auto orig_to_proto = [&](float v, float pad, float s) {
        return (v * lb.scale + pad) * s;
    };

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
        int x1 = (int)std::floor(orig_to_proto(box.x1, lb.pad_left, sx));
        int y1 = (int)std::floor(orig_to_proto(box.y1, lb.pad_top,  sy));
        int x2 = (int)std::ceil (orig_to_proto(box.x2, lb.pad_left, sx));
        int y2 = (int)std::ceil (orig_to_proto(box.y2, lb.pad_top,  sy));

        x1 = std::max(0, std::min(x1, proto_w - 1));
        y1 = std::max(0, std::min(y1, proto_h - 1));
        x2 = std::max(0, std::min(x2, proto_w - 1));
        y2 = std::max(0, std::min(y2, proto_h - 1));

        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);

        for (int h = 0; h < proto_h; h++) {
            for (int w = 0; w < proto_w; w++) {
                if (w < x1 || w > x2 || h < y1 || h > y2) {
                    mask.data[(size_t)h * proto_w + w] = 0.0f;
                }
            }
        }

        // 缩放到原图尺寸并二值化
        // 两步走，不能直接把 proto 网格拉到原图尺寸：
        //   proto(160x160) → 模型输入(input_w x input_h) → 去letterbox → 原图
        // 直接 proto → orig 会忽略 letterbox 的 padding，把掩码横向/纵向拉伸，
        // 实测掩码 IoU 仅 0.24（bus.jpg 640，yolo11s-seg）。
        cv::Mat mask_cv(proto_h, proto_w, CV_32FC1, mask.data.data());
        cv::Mat to_input, to_orig;
        cv::resize(mask_cv, to_input, cv::Size(input_width, input_height));

        // 去掉 letterbox padding：按 scale 裁掉上下/左右边框后再缩放到原图
        const int crop_w = std::min(input_width,  (int)std::lround(orig_w * lb.scale));
        const int crop_h = std::min(input_height, (int)std::lround(orig_h * lb.scale));
        const int off_x = (input_width  - crop_w) / 2;
        const int off_y = (input_height - crop_h) / 2;
        if (crop_w > 0 && crop_h > 0) {
            cv::Mat cropped = to_input(cv::Rect(off_x, off_y, crop_w, crop_h));
            cv::resize(cropped, to_orig, cv::Size(orig_w, orig_h));
        } else {
            cv::resize(to_input, to_orig, cv::Size(orig_w, orig_h));
        }

        cv::Mat binary;
        cv::threshold(to_orig, binary, 0.5f, 1.0f, cv::THRESH_BINARY);

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
                                     ctx_.input_width, ctx_.input_height);
    }
    return result;
}

} // namespace yolo_onnx