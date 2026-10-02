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
    ///
    /// 模型差异全部由 PostProcess + Decoder 承担，本类不再为每个模型派生子类。
    virtual InferResult infer(const cv::Mat& image);

    /// 类型化便捷入口（供需要确定返回类型的调用方，如 Python 绑定）
    DetectResult  infer_detect(const cv::Mat& image);
    SegmentResult infer_segment(const cv::Mat& image);
    PoseResult    infer_pose(const cv::Mat& image);
    OBBResult     infer_obb(const cv::Mat& image);

    /// 由工厂设置模型类型与任务类型（在 load 之前调用）。
    /// 模型不再按版本派生子类，类型信息全部承载在 config_ 中。
    void set_model_task(ModelType type, TaskType task) {
        config_.model_type = type;
        config_.task_type  = task;
    }

    /// Get config
    const Config& config() const { return config_; }

    /// Get backend
    std::shared_ptr<Backend> backend() const { return backend_; }

protected:
    Config                config_;
    std::shared_ptr<Backend> backend_;
    mutable std::shared_ptr<PreProcess>  preprocess_;
    std::shared_ptr<PostProcess>         postprocess_;

    /// 构造本模型的预处理参数（默认按 model_type 生成）
    virtual PreProcessParams make_preprocess_params() const {
        return PreProcessParams::for_model(config_.model_type,
                                           config_.input_width, config_.input_height);
    }

    /// 构造本模型的后处理参数（默认取 config_ 的 score/nms 阈值）
    virtual PostProcessParams make_postprocess_params() const {
        PostProcessParams p;
        p.score_thresh = config_.score_thresh;
        p.nms_thresh   = config_.nms_thresh;
        return p;
    }

    /// Letterbox resize + normalize（复用独立的 PreProcess 模块）
    PreProcessResult preprocess(const cv::Mat& image, int target_w, int target_h) const;

    /// 统一流程：预处理 → 前向 → PostProcess（解码+还原+NMS 全在后处理内完成）
    InferResult run_pipeline(const cv::Mat& image);
};

// ============================================================
// Factory: create a model by type
// ============================================================
std::shared_ptr<Model> create_model(ModelType type);
std::shared_ptr<Model> create_model(ModelType type, TaskType task);

} // namespace yolo_onnx