#pragma once
#include "yolo_onnx/yolo_onnx_types.hpp"

#include <vector>

namespace yolo_onnx {

// ============================================================
// PostProcess — 独立的后处理模块
// ============================================================
// 职责：坐标还原（letterbox 逆映射）、置信度过滤、NMS。
// 与模型解耦：decode_output 只负责把模型原始输出解码为
// BoxArray / OBBBoxArray（模型输入尺寸坐标系），其余统一交给这里。
// ============================================================

/// 对解码后的框做坐标还原 + NMS，返回最终结果
DetectResult postprocess_detect(BoxArray boxes, const LetterboxInfo& letterbox,
                                float nms_iou);

/// OBB 版本：坐标还原 + 旋转 IoU NMS
OBBResult postprocess_obb(OBBBoxArray boxes, const LetterboxInfo& letterbox,
                          float nms_iou);

/// IoU / NMS 基础工具（坐标还原 scale_box/scale_keypoint/scale_obb_box 见 yolo_onnx_types.hpp）
float iou(const Box& a, const Box& b);
std::vector<int> nms(const BoxArray& boxes, float iou_threshold);

/// OBB 旋转 IoU 与 NMS
float obb_iou(const OBBBox& a, const OBBBox& b);
std::vector<int> obb_nms(const OBBBoxArray& boxes, float iou_threshold);

} // namespace yolo_onnx
