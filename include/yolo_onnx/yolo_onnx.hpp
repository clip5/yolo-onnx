#pragma once

// ============================================================
// yolo_onnx.hpp — 唯一对外公共接口头文件
// ============================================================
// 只需包含此一个头文件即可使用 yolo_onnx 库的全部公共 API。
// 只依赖 yolo_onnx_types.hpp（核心类型定义），不依赖 backend / model 等内部头文件。
//
// 快速开始：
//   #include "yolo_onnx/yolo_onnx.hpp"
//
//   auto model = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8);
//   model->load({...});
//   auto boxes = model->infer(image);
//
// 多任务推理（segment / pose / OBB）：
//   auto result = model->infer(image);
//   if (auto* seg = std::get_if<yolo_onnx::SegmentResult>(&result)) {
//       // seg->boxes, seg->masks
//   }
// ============================================================

#include "yolo_onnx/yolo_onnx_types.hpp"

#include <memory>
#include <opencv2/core.hpp>

namespace yolo_onnx {

// ============================================================
// Inference Backend Interface (前置声明)
// ============================================================
// Backend 是内部实现细节，对外接口只需前置声明。
// 完整定义在 backend.hpp 中，用户无需包含。
// ============================================================
class Backend;

// ============================================================
// YOLO Model Interface
// ============================================================
// 每个 YOLO 变体实现自己的 pre/post 处理逻辑。
// 通过工厂函数 create_model() 创建实例。
// ============================================================

class Model {
public:
    /// 模型配置
    struct Config {
        std::string model_path;        // Path to ONNX model
        ModelType   model_type   = ModelType::YOLOv8;
        TaskType    task_type    = TaskType::Detect;
        std::string backend_type = "onnxruntime";  // backend name
        float       score_thresh = 0.5f;
        float       nms_thresh   = 0.45f;
        int         input_width  = 640;
        int         input_height = 640;
        int         num_classes  = 80;
        int         num_keypoints = 17;  // for pose (COCO default)
        int         num_threads  = 4;   // CPU threads for onnxruntime
        std::string custom_config;      // Backend-specific config (key=value;key=value...)
    };

    virtual ~Model() = default;

    /// Load model and initialize backend
    virtual bool load(const Config& config);

    /// Run inference on a single image. Returns the full task-specific result.
    /// Use std::get_if<> / std::holds_alternative<> to extract the result:
    ///   auto result = model->infer(image);
    ///   if (auto* det = std::get_if<DetectResult>(&result)) { ... }   // boxes
    ///   if (auto* seg = std::get_if<SegmentResult>(&result)) { ... } // boxes + masks
    ///   if (auto* pose = std::get_if<PoseResult>(&result)) { ... }   // boxes + keypoints
    ///   if (auto* obb = std::get_if<OBBResult>(&result)) { ... }     // obb_boxes
    virtual InferResult infer(const cv::Mat& image) = 0;

    /// Get config
    const Config& config() const { return config_; }

    /// Get backend
    std::shared_ptr<Backend> backend() const { return backend_; }

protected:
    Config                config_;
    std::shared_ptr<Backend> backend_;

    /// Letterbox resize + normalize
    PreProcessResult preprocess(const cv::Mat& image, int target_w, int target_h) const;

    /// Per-channel normalization applied after the default /255 scaling.
    /// `ch`: 0=R, 1=G, 2=B. Default: identity (values stay in [0,1]).
    /// Override for models that expect e.g. ImageNet mean/std (YOLOX).
    virtual float normalize_channel(float v, int ch) const { return v; }

    /// Decode model output into candidate boxes (model-specific)
    virtual BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const = 0;
};

// ============================================================
// Factory: create a model by type
// ============================================================
std::shared_ptr<Model> create_model(ModelType type);
std::shared_ptr<Model> create_model(ModelType type, TaskType task);

// ============================================================
// Task-specific model forward declarations
// ============================================================
class ModelV8Segment;
class ModelV8Pose;
class ModelV8OBB;

} // namespace yolo_onnx