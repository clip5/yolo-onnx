#include "process/postprocess/postprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace yolo_onnx {

// ============================================================
// PostProcessSem — 语义分割
// ============================================================
// 模型输出 [1, H, W] 的**类别 id 图**（ultralytics 导出末端为
// Resize → ArgMax → Cast(uint8)），即 argmax 已在图内完成，后端把它
// 转成 float32 承载（类别 id < 2^24，float32 无损）。
//
// 与 detect/segment/pose/obb 的结构性差异：
//   * 没有候选框 → 不走 Decoder，也不需要 NMS / 置信度过滤 / top_k
//   * 输出是稠密图 → 唯一的「还原」动作是去掉 letterbox padding 并缩回原图
//
// 坐标还原与 mask 的做法同源（见 segment.cpp 的说明）：类别图对应的是
// **模型输入画布**，画布上图像内容占据 (pad_top, pad_left) 起的新_w×new_h
// 区域，其余是 114 灰边。因此先按 letterbox 比例裁掉灰边，再用**最近邻**
// 缩放到原图尺寸——类别 id 必须用最近邻，双线性会在类别边界产生不存在的 id。

int PostProcessSem::find_classmap_index(const TensorSet& outputs) {
    int best = -1;
    int64_t best_elems = -1;
    for (size_t i = 0; i < outputs.size(); i++) {
        const auto& shape = outputs.shape(i);
        // 类别图：[1, H, W]（3D）或 [H, W] / [1, H, W, 1] 等等价写法
        if (shape.size() < 2) continue;
        int64_t elems = 1;
        for (auto d : shape) elems *= d;
        if (elems > best_elems) {
            best_elems = elems;
            best = (int)i;
        }
    }
    return best;
}

InferResult PostProcessSem::forward(const TensorSet& outputs,
                                    const LetterboxInfo& lb) const {
    const int idx = find_classmap_index(outputs);
    if (idx < 0) return SemResult{};

    const auto& shape = outputs.shape(idx);
    const float* data = outputs.data(idx);
    if (!data) return SemResult{};

    // ---- 归一成 [map_h, map_w] ----
    // 常规导出为 [1, H, W]；其余等价形状按「元素总数相同即同一张图」处理，
    // 取最后两维作为 H/W（[H, W] / [1, H, W, 1] 都能落到这一支）。
    int map_h = 0, map_w = 0;
    if (shape.size() == 3 && shape[0] == 1) {
        map_h = (int)shape[1];
        map_w = (int)shape[2];
    } else if (shape.size() == 2) {
        map_h = (int)shape[0];
        map_w = (int)shape[1];
    } else {
        // 兜底：由元素总数与较短边推断正方形网格（仅在非标准导出时触发）
        int64_t total = 1;
        for (auto d : shape) total *= d;
        int side = (int)std::lround(std::sqrt((double)total));
        if (side <= 0 || (int64_t)side * side != total) {
            std::cerr << "[PostProcessSem] Unexpected class-map shape: ";
            for (auto d : shape) std::cerr << d << " ";
            std::cerr << std::endl;
            return SemResult{};
        }
        map_h = map_w = side;
    }
    if (map_h <= 0 || map_w <= 0) return SemResult{};

    // ---- 裁掉 letterbox padding ----
    // 类别图是模型输入画布的下采样版本，故画布坐标 → 类别图坐标的比例为
    // map_w / target_w。内容区域 = 新_w×新_h 映射到类别图上的矩形。
    const int target_w = lb.target_w > 0 ? lb.target_w : ctx_.input_width;
    const int target_h = lb.target_h > 0 ? lb.target_h : ctx_.input_height;
    if (target_w <= 0 || target_h <= 0) return SemResult{};

    const int content_w = std::min(target_w,  (int)std::lround(lb.orig_w * lb.scale));
    const int content_h = std::min(target_h, (int)std::lround(lb.orig_h * lb.scale));
    const int off_x = std::max(0, (target_w  - content_w) / 2);
    const int off_y = std::max(0, (target_h - content_h) / 2);

    int crop_x = (int)std::lround((double)off_x * map_w / target_w);
    int crop_y = (int)std::lround((double)off_y * map_h / target_h);
    int crop_w = (int)std::lround((double)content_w * map_w / target_w);
    int crop_h = (int)std::lround((double)content_h * map_h / target_h);

    crop_w = std::min(crop_w, map_w - crop_x);
    crop_h = std::min(crop_h, map_h - crop_y);
    if (crop_w <= 0 || crop_h <= 0) {
        // 退化情况（模型输出比 padding 还小）：退回整张图，至少给出可用结果
        crop_x = crop_y = 0;
        crop_w = map_w;
        crop_h = map_h;
    }

    // ---- 缩回原图尺寸（最近邻：类别 id 不能插值） ----
    // 后端已把 uint8 类别图转成 float32，这里用 CV_32FC1 承载后按最近邻缩放。
    cv::Mat map_cv(map_h, map_w, CV_32FC1, const_cast<float*>(data));
    cv::Mat cropped = map_cv(cv::Rect(crop_x, crop_y, crop_w, crop_h));

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(lb.orig_w, lb.orig_h), 0, 0, cv::INTER_NEAREST);

    SemResult result;
    result.mask = Mask(lb.orig_w, lb.orig_h);
    std::memcpy(result.mask.data.data(), resized.ptr<float>(),
                (size_t)lb.orig_w * lb.orig_h * sizeof(float));
    return result;
}

} // namespace yolo_onnx