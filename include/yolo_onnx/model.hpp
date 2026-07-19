#pragma once
#include "yolo_onnx/types.hpp"
#include "yolo_onnx/backend.hpp"
#include <memory>
#include <opencv2/core.hpp>

namespace yolo_onnx {

// ============================================================
// YOLO Model Interface
// ============================================================
// Each YOLO variant implements its own pre/post processing.
// ============================================================

class Model {
public:
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
    };

    virtual ~Model() = default;

    /// Load model and initialize backend
    virtual bool load(const Config& config);

    /// Run inference on a single image (detection)
    virtual BoxArray infer(const cv::Mat& image) = 0;

    /// Get config
    const Config& config() const { return config_; }

    /// Get backend
    std::shared_ptr<Backend> backend() const { return backend_; }

protected:
    Config                config_;
    std::shared_ptr<Backend> backend_;

    /// Letterbox resize + normalize
    PreProcessResult preprocess(const cv::Mat& image, int target_w, int target_h) const;

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