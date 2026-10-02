#pragma once
#include <vector>
#include <string>
#include <memory>
#include <variant>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace yolo_onnx {

// ============================================================
// Model type enum
// ============================================================
enum class ModelType {
    YOLOv5   = 0,
    YOLOX    = 1,
    YOLOv8   = 2,
    YOLOv11  = 3,
    YOLO26   = 4,
    PPYOLOE  = 5,
};

inline const char* model_type_name(ModelType type) {
    switch (type) {
        case ModelType::YOLOv5:   return "YOLOv5";
        case ModelType::YOLOX:    return "YOLOX";
        case ModelType::YOLOv8:   return "YOLOv8";
        case ModelType::YOLOv11:  return "YOLOv11";
        case ModelType::YOLO26:   return "YOLO26";
        case ModelType::PPYOLOE:  return "PPYOLOE";
        default:                  return "Unknown";
    }
}

// ============================================================
// Task type enum
// ============================================================
enum class TaskType {
    Detect   = 0,   // Object detection
    Segment  = 1,   // Instance segmentation
    Pose     = 2,   // Pose/keypoint estimation
    OBB      = 3,   // Oriented bounding box
};

inline const char* task_type_name(TaskType type) {
    switch (type) {
        case TaskType::Detect:   return "detect";
        case TaskType::Segment:  return "segment";
        case TaskType::Pose:     return "pose";
        case TaskType::OBB:      return "obb";
        default:                 return "unknown";
    }
}

// ============================================================
// Detection Box
// ============================================================
struct Box {
    float x1{}, y1{}, x2{}, y2{};  // corner coordinates (in original image)
    float score{};
    int   label{};

    Box() = default;
    Box(float x1_, float y1_, float x2_, float y2_, float score_, int label_)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(score_), label(label_) {}

    float width()  const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area()   const { return width() * height(); }
};

using BoxArray = std::vector<Box>;

// ============================================================
// Keypoint (for Pose estimation)
// ============================================================
struct Keypoint {
    float x{}, y{};          // coordinate in original image
    float visibility{};      // 0=not labeled, 1=labeled not visible, 2=labeled and visible
};

// ============================================================
// Mask (for Segmentation)
// ============================================================
struct Mask {
    int width{}, height{};       // mask dimensions
    std::vector<float> data;     // HxW float mask, values in [0, 1]

    Mask() = default;
    Mask(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}
};

// ============================================================
// Oriented Bounding Box (for OBB)
// ============================================================
struct OBBBox {
    float cx{}, cy{};    // center
    float w{}, h{};      // width, height
    float angle{};       // rotation angle in radians
    float score{};
    int   label{};

    OBBBox() = default;
    OBBBox(float cx_, float cy_, float w_, float h_, float angle_,
           float score_, int label_)
        : cx(cx_), cy(cy_), w(w_), h(h_), angle(angle_),
          score(score_), label(label_) {}

    /// Get the 4 corner points of the rotated box
    std::vector<cv::Point2f> corners() const {
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float hw = w / 2.0f;
        float hh = h / 2.0f;

        std::vector<cv::Point2f> pts(4);
        pts[0] = cv::Point2f(cx - hw * cos_a + hh * sin_a,
                             cy - hw * sin_a - hh * cos_a);
        pts[1] = cv::Point2f(cx + hw * cos_a + hh * sin_a,
                             cy + hw * sin_a - hh * cos_a);
        pts[2] = cv::Point2f(cx + hw * cos_a - hh * sin_a,
                             cy + hw * sin_a + hh * cos_a);
        pts[3] = cv::Point2f(cx - hw * cos_a - hh * sin_a,
                             cy - hw * sin_a + hh * cos_a);
        return pts;
    }

    /// Get axis-aligned bounding box
    Box aabb() const {
        auto pts = corners();
        float min_x = pts[0].x, min_y = pts[0].y;
        float max_x = pts[0].x, max_y = pts[0].y;
        for (int i = 1; i < 4; i++) {
            min_x = std::min(min_x, pts[i].x);
            min_y = std::min(min_y, pts[i].y);
            max_x = std::max(max_x, pts[i].x);
            max_y = std::max(max_y, pts[i].y);
        }
        return Box(min_x, min_y, max_x, max_y, score, label);
    }
};

using OBBBoxArray = std::vector<OBBBox>;

// ============================================================
// Result types for each task
// ============================================================
struct DetectResult {
    BoxArray boxes;
};

struct SegmentResult {
    BoxArray        boxes;
    std::vector<Mask> masks;
};

struct PoseResult {
    BoxArray               boxes;
    std::vector<std::vector<Keypoint>> keypoints;  // one per box
};

struct OBBResult {
    OBBBoxArray obb_boxes;
};

// ============================================================
// Unified inference result variant (supports all task types)
// ============================================================
using InferResult = std::variant<DetectResult, SegmentResult, PoseResult, OBBResult>;

// ============================================================
// Letterbox transform info (for mapping back to original image)
// ============================================================
struct LetterboxInfo {
    float scale;       // resize scale factor
    int   pad_left;    // padding on left
    int   pad_top;     // padding on top
    int   orig_w;      // original image width
    int   orig_h;      // original image height
    int   target_w;    // target image width (after letterbox)
    int   target_h;    // target image height (after letterbox)
};

// ============================================================
// Pre-processing result
// ============================================================
struct PreProcessResult {
    std::vector<float> blob;         // NCHW float blob
    LetterboxInfo      letterbox;    // letterbox info for coordinate mapping
};

// ============================================================
// 预处理参数（描述一个模型对输入张量的要求）
// ============================================================
struct PreProcessParams {
    int   target_width   = 640;
    int   target_height  = 640;
    bool  swap_rb        = true;    // BGR→RGB（PPYOLOE 等保持 BGR 时为 false）
    float scale_factor   = 1.0f / 255.0f;  // 像素缩放
    // 可选的每通道 mean/std（在 scale_factor 之后应用: (v*scale - mean) / std）
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3]  = {1.0f, 1.0f, 1.0f};

    /// 按模型类型给出默认参数（实现在 src/process/preprocess/preprocess.cpp）
    static PreProcessParams for_model(ModelType type, int width, int height);
};

// ============================================================
// Utility functions
// ============================================================
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// IoU / NMS 实现在 src/process/postprocess/postprocess.cpp（模块内部使用，
// 但属于对外工具 API，故在此声明）

/// 两框 IoU
float iou(const Box& a, const Box& b);

/// CPU NMS（for regular boxes），返回保留框的索引
std::vector<int> nms(const BoxArray& boxes, float iou_threshold);

/// Scale box from model output space back to original image
inline void scale_box(Box& box, const LetterboxInfo& info) {
    box.x1 = (box.x1 - info.pad_left) / info.scale;
    box.y1 = (box.y1 - info.pad_top) / info.scale;
    box.x2 = (box.x2 - info.pad_left) / info.scale;
    box.y2 = (box.y2 - info.pad_top) / info.scale;
    // Clamp to original image bounds
    box.x1 = std::max(0.0f, std::min(box.x1, (float)info.orig_w));
    box.y1 = std::max(0.0f, std::min(box.y1, (float)info.orig_h));
    box.x2 = std::max(0.0f, std::min(box.x2, (float)info.orig_w));
    box.y2 = std::max(0.0f, std::min(box.y2, (float)info.orig_h));
}

/// Scale keypoint from model output space back to original image
inline void scale_keypoint(Keypoint& kp, const LetterboxInfo& info) {
    kp.x = (kp.x - info.pad_left) / info.scale;
    kp.y = (kp.y - info.pad_top) / info.scale;
    kp.x = std::max(0.0f, std::min(kp.x, (float)info.orig_w));
    kp.y = std::max(0.0f, std::min(kp.y, (float)info.orig_h));
}

/// Scale OBB box from model output space back to original image
inline void scale_obb_box(OBBBox& box, const LetterboxInfo& info) {
    box.cx = (box.cx - info.pad_left) / info.scale;
    box.cy = (box.cy - info.pad_top) / info.scale;
    box.w  = box.w / info.scale;
    box.h  = box.h / info.scale;
}

} // namespace yolo_onnx