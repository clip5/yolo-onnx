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
// blob 以 cv::Mat 承载 NCHW float32 张量（dims=4, size={1,3,H,W}），
// 这样形状信息自描述、无需额外传 shape，且内存连续可直接交给推理后端。
// 注意：NCHW 是 4 维数据，cv::Mat 的原生维度只能表达 HWC 交织，
// 这里通过显式 steps 强制为平面序（见 make_nchw_blob）。
struct PreProcessResult {
    cv::Mat       blob;       // NCHW float32: dims=4, size={1,3,target_h,target_w}, 连续
    LetterboxInfo letterbox;  // letterbox info for coordinate mapping

    /// NCHW 形状 {1, 3, H, W}，供推理后端直接使用
    std::vector<int64_t> shape() const {
        if (blob.empty()) return {};
        return {1, blob.size[1], blob.size[2], blob.size[3]};
    }

    /// 连续 float 裸指针（NCHW），供需要裸缓冲的后端使用
    const float* data() const { return blob.empty() ? nullptr : blob.ptr<float>(); }

    size_t total_bytes() const { return blob.empty() ? 0 : blob.total() * blob.elemSize(); }
};

/// 分配一个 NCHW 连续 float32 Mat（持有内存）。
/// cv::Mat 的原生维度只能表达 HWC 交织，但当 4 维 Mat 自行分配连续内存时，
/// 其 steps 恰好就是 NCHW 平面序（step[1]=H*W*4, step[2]=W*4, step[3]=4），
/// 因此无需手动指定 steps。
inline cv::Mat make_nchw_blob(int batch, int channels, int height, int width) {
    int sizes[4] = {batch, channels, height, width};
    return cv::Mat(4, sizes, CV_32F);
}

// ============================================================
// 预处理参数（描述一个模型对输入张量的要求）
// ============================================================
struct PreProcessParams {
    ModelType model_type    = ModelType::YOLOv8;  // 记录来源，便于参数缓存失效判断
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
// NMS 策略
// ============================================================
// ClassAware：不同类别的框之间互不抑制（YOLOv5/v8/v11 等默认行为，
//            类分数与框坐标一同按类别分组后各自做 NMS）
// Agnostic ：所有类别的框一起做 NMS（早期 YOLOv3/v4 的做法，
//            跨类重叠的框会互相抑制）
enum class NmsMode {
    ClassAware = 0,
    Agnostic   = 1,
};

// ============================================================
// 后处理参数
// ============================================================
struct PostProcessParams {
    float score_thresh   = 0.25f;  // 置信度下限
    float nms_thresh     = 0.45f;  // NMS IoU 阈值
    int   max_detections = 0;      // 最多保留目标数，0 表示不限制
    NmsMode nms_mode     = NmsMode::ClassAware;  // NMS 是否按类别分组
};

// ============================================================
// 解码上下文：Decoder 解码所需的模型配置（与具体输出格式无关的部分）
// ============================================================
struct DecodeContext {
    int   num_classes   = 80;
    int   num_keypoints = 17;  // pose 用
    float score_thresh  = 0.25f;
    int   input_width   = 640;
    int   input_height  = 640;

    /// 第 level 个特征层的步长（level 0/1/2 → stride 8/16/32）
    int stride(int level) const { return 8 << level; }

    /// 第 level 个特征层的网格宽/高/格点数。
    /// 必须由实际输入尺寸推导：硬编码 80/40/20 与 6400/1600/400 会对
    /// 非正方形输入（如 640x352）解出错误网格并读越界。
    int grid_w(int level) const { return input_width  / stride(level); }
    int grid_h(int level) const { return input_height / stride(level); }
    int grid_count(int level) const { return grid_w(level) * grid_h(level); }
};

// ============================================================
// TensorSet — 推理后端的一组输出张量
// ============================================================
// 每个 datas[i] 是一个连续的 1-D float32 Mat（元素总数 = shapes[i] 各维乘积），
// 形状自描述地保存在 shapes[i] 中。相比裸 vector<vector<float>>，用 Mat 承载
// 便于与预处理产出的 blob 保持同一种类型，也便于后续替换/扩展张量类型。
struct TensorSet {
    std::vector<cv::Mat>              datas;    // 连续 float32 数据
    std::vector<std::vector<int64_t>> shapes;   // 与 datas 一一对应的形状

    size_t size() const { return datas.size(); }
    bool   empty() const { return datas.empty(); }

    /// 第 i 个张量的元素总数（未越界时为 0）
    size_t count(size_t i) const {
        if (i >= datas.size()) return 0;
        return datas[i].total() * datas[i].channels();
    }

    /// 第 i 个张量的形状，缺省为空 vector
    const std::vector<int64_t>& shape(size_t i) const {
        static const std::vector<int64_t> kEmpty;
        return i < shapes.size() ? shapes[i] : kEmpty;
    }

    /// 第 i 个张量的裸数据指针（未越界时为 nullptr）
    const float* data(size_t i) const {
        if (i >= datas.size() || datas[i].empty()) return nullptr;
        return datas[i].ptr<float>();
    }
};

// ============================================================
// Utility functions
// ============================================================
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 注：IoU / NMS / 坐标还原 / 过滤 / 一站式 pipeline 等后处理函数声明在
//     src/process/postprocess/postprocess_core.hpp（不依赖 Model/Backend/Decoder，
//     可整份拷到其他项目复用）。此处仅保留无需依赖后处理的纯类型与内联工具。

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