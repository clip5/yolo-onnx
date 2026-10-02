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
// Backend / PreProcess / PostProcess 是内部实现细节，对外接口只需前置声明。
// 完整定义在 src/core/backend.hpp、src/process/preprocess/preprocess.hpp、
// src/process/postprocess/postprocess.hpp 中，用户无需包含。
// ============================================================
class Backend;
class PreProcess;
class PostProcess;

namespace detail { struct ModelImpl; }

// ============================================================
// YOLO Model Interface
// ============================================================
// 所有模型共用这一个类：模型格式与任务差异全部由内部 PostProcess + Decoder
// 承担，通过 Config 表达。实现细节隐藏于 detail::ModelImpl（PIMPL），
// 本类对外只暴露配置与推理入口。
//
// 典型用法：
//   auto model = yolo_onnx::create_model(ModelType::YOLOv8, TaskType::Detect);
//   model->load(cfg);
//   InferResult r = model->infer(image);       // 或 model->infer_detect(image)
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

    explicit Model();
    ~Model();

    Model(Model&&) noexcept;             // 支持移动（unique_ptr 成员）
    Model& operator=(Model&&) noexcept;
    Model(const Model&)            = delete;   // 内部持有独占资源，不可拷贝
    Model& operator=(const Model&) = delete;

    /// Load model and initialize backend
    bool load(const Config& config);

    /// Run inference. Result type follows Config::task_type:
    ///   if (auto* det = std::get_if<DetectResult>(&r))  { ... }  // boxes
    ///   if (auto* seg = std::get_if<SegmentResult>(&r)) { ... }  // boxes + masks
    ///   if (auto* pose = std::get_if<PoseResult>(&r))  { ... }  // boxes + keypoints
    ///   if (auto* obb = std::get_if<OBBResult>(&r))    { ... }  // obb_boxes
    InferResult infer(const cv::Mat& image);

    /// 类型化入口（需要确定返回类型时使用，避免 std::variant 分支判断）
    DetectResult  infer_detect(const cv::Mat& image);
    SegmentResult infer_segment(const cv::Mat& image);
    PoseResult    infer_pose(const cv::Mat& image);
    OBBResult     infer_obb(const cv::Mat& image);

    /// 由工厂设置模型类型与任务类型（load 之前调用）
    void set_model_task(ModelType type, TaskType task);

    /// 当前配置（load 后可能包含模型实际输入尺寸）
    const Config& config() const;

    /// 推理后端实例（未load 时为空）
    std::shared_ptr<Backend> backend() const;

private:
    Config                     config_;
    std::unique_ptr<detail::ModelImpl> impl_;
};

// ============================================================
// Factory: create a model by type
// ============================================================
std::shared_ptr<Model> create_model(ModelType type);
std::shared_ptr<Model> create_model(ModelType type, TaskType task);

} // namespace yolo_onnx