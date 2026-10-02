#pragma once
// ============================================================
// postprocess_core.hpp — 后处理核心（可独立复用的最小接口）
// ============================================================
// 本文件只依赖公共类型 yolo_onnx_types.hpp（纯POD + cv::Mat），不依赖
// Model / Backend / Decoder，也不依赖 yolo_onnx.hpp 之外任何工程内部头。
// 因此可以整份拷到其他项目直接使用——只需要 Box/OBBBox/LetterboxInfo 等
// 数据类型定义即可（同样在本项目内是无依赖的 header-only 类型）。
//
// 提供三层能力，从细到粗：
//   1) 基元    ：iou / nms / nms_class_aware / obb_iou / obb_nms
//   2) 步骤    ：restore_*（坐标还原）/ filter_by_score / top_k
//   3) 组合    ：detect_pipeline / obb_pipeline / select_indices
//
// 最简用法（只需一个后处理调用）：
//   auto boxes = yolo_onnx::detect_pipeline(candidates, letterbox, params);
//
// 全部函数都是自由函数，无对象状态、无隐式依赖，可自由挑选组合。
// ============================================================

#include "yolo_onnx/yolo_onnx_types.hpp"

#include <vector>

namespace yolo_onnx {

// ============================================================
// 1) 基元：IoU / NMS
// ============================================================

/// 轴对齐框 IoU
float iou(const Box& a, const Box& b);

/// NMS。@param mode ClassAware（默认，不同类互不抑制）或 Agnostic（跨类抑制）
std::vector<int> nms(const BoxArray& boxes, float iou_threshold,
                     NmsMode mode = NmsMode::ClassAware);

/// 仅 class-aware 版本（等价于 nms(..., NmsMode::ClassAware)）
std::vector<int> nms_class_aware(const BoxArray& boxes, float iou_threshold);

/// 旋转框 IoU 与 NMS（始终按类别分组：旋转框本身类别明确，无需两种模式）
float obb_iou(const OBBBox& a, const OBBBox& b);
std::vector<int> obb_nms(const OBBBoxArray& boxes, float iou_threshold);

// ============================================================
// 2) 步骤：坐标还原 / 过滤 / 截断
// ============================================================

/// 坐标还原（letterbox 逆映射），in-place，不改动 score / label
void restore_boxes(BoxArray& boxes, const LetterboxInfo& lb);
void restore_keypoints(std::vector<Keypoint>& kps, const LetterboxInfo& lb);
void restore_obb(OBBBoxArray& boxes, const LetterboxInfo& lb);

/// 置信度过滤（@param score_thresh <= 0 表示不过滤）
BoxArray    filter_by_score(const BoxArray& boxes, float score_thresh);
OBBBoxArray filter_by_score(const OBBBoxArray& boxes, float score_thresh);

/// 保留分数最高的 @p max_detections 个（<= 0 表示不限制）
BoxArray    top_k(BoxArray boxes, int max_detections);
OBBBoxArray top_k(OBBBoxArray boxes, int max_detections);

// ============================================================
// 3) 组合：一站式后处理
// ============================================================

/// 检测全流程：过滤 → 坐标还原 → NMS → 截断
/// @param candidates 已解码的候选框（模型输入尺寸坐标系）
/// @param lb          letterbox 信息，用于还原到原图坐标系
/// @param params      阈值与 NMS 策略
/// @return 原图坐标系的最终框（按分数降序）
BoxArray detect_pipeline(BoxArray candidates, const LetterboxInfo& lb,
                         const PostProcessParams& params);

/// 旋转框全流程：过滤 → 坐标还原 → 旋转 NMS → 截断
OBBBoxArray obb_pipeline(OBBBoxArray candidates, const LetterboxInfo& lb,
                         const PostProcessParams& params);

/// 过滤 → 坐标还原 → NMS → 截断，返回**保留的候选下标**（而非框本身）。
/// segment / pose 需要按下标回取 mask 系数 / 关键点，故必须走这个入口。
/// 下标按分数降序排列。
std::vector<int> select_indices(const std::vector<Box>& candidates,
                                const LetterboxInfo& lb,
                                const PostProcessParams& params);

} // namespace yolo_onnx