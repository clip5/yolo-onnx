#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <cmath>

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
// Utility functions
// ============================================================
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float iou(const Box& a, const Box& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// CPU NMS
inline std::vector<int> nms(const BoxArray& boxes, float iou_threshold) {
    std::vector<int> indices;
    std::vector<int> order(boxes.size());
    for (int i = 0; i < (int)boxes.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return boxes[i].score > boxes[j].score;
    });
    std::vector<bool> removed(boxes.size(), false);
    for (int i = 0; i < (int)order.size(); i++) {
        if (removed[order[i]]) continue;
        indices.push_back(order[i]);
        for (int j = i + 1; j < (int)order.size(); j++) {
            if (removed[order[j]]) continue;
            if (iou(boxes[order[i]], boxes[order[j]]) > iou_threshold) {
                removed[order[j]] = true;
            }
        }
    }
    return indices;
}

// Map box from model output space back to original image
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

} // namespace yolo_onnx