#include "yolo_onnx/yolo_onnx.hpp"
#include "core/onnxruntime_backend.hpp"
#include "process/preprocess/preprocess.hpp"
#include "process/postprocess/postprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace yolo_onnx {

bool Model::load(const Config& config) {
    config_ = config;

    // 1. Create backend
    backend_ = create_backend(config.backend_type);
    if (!backend_) {
        std::cerr << "[Model] Failed to create backend: " << config.backend_type << std::endl;
        return false;
    }

    // 2. Load model
    Backend::Config backend_config;
    backend_config.model_path   = config.model_path;
    backend_config.num_threads  = config.num_threads;
    backend_config.device_id    = 0;
    backend_config.enable_fp16  = false;
    backend_config.enable_int8  = false;
    backend_config.custom_config = config.custom_config;

    // Parse custom_config for device_id and fp16
    if (!config.custom_config.empty()) {
        auto parse_kv = [&](const std::string& kv) {
            if (kv.find("--device=") == 0) {
                backend_config.device_id = std::stoi(kv.substr(9));
            } else if (kv.find("--fp16") == 0) {
                backend_config.enable_fp16 = true;
            } else if (kv.find("--int8") == 0) {
                backend_config.enable_int8 = true;
            }
        };
        size_t start = 0, end;
        while ((end = config.custom_config.find(';', start)) != std::string::npos) {
            parse_kv(config.custom_config.substr(start, end - start));
            start = end + 1;
        }
        if (start < config.custom_config.size()) {
            parse_kv(config.custom_config.substr(start));
        }
    }

    if (!backend_->load(backend_config)) {
        std::cerr << "[Model] Failed to load model: " << config.model_path << std::endl;
        return false;
    }

    // 3. Override input dimensions from model if available
    auto input_shapes = backend_->get_input_shapes();
    if (!input_shapes.empty() && input_shapes[0].size() >= 4) {
        if (input_shapes[0][2] > 0) config_.input_height = (int)input_shapes[0][2];
        if (input_shapes[0][3] > 0) config_.input_width  = (int)input_shapes[0][3];
    }

    std::cout << "[" << model_type_name(config_.model_type) 
              << "] Model loaded: " << config_.model_path
              << " (" << backend_->name() << ")"
              << " input: " << config_.input_width << "x" << config_.input_height
              << std::endl;

    return true;
}

PreProcessResult Model::preprocess(const cv::Mat& image, int target_w, int target_h) const {
    // 复用独立的 PreProcess 模块（懒初始化，跟随最新 config_ 的输入尺寸）
    if (!preprocess_ ||
        preprocess_->params().target_width != target_w ||
        preprocess_->params().target_height != target_h) {
        auto params = make_preprocess_params();
        params.target_width = target_w;
        params.target_height = target_h;
        preprocess_ = std::make_shared<PreProcess>(params);
    }
    return preprocess_->run(image);
}

// ============================================================
// 统一推理流程
// ============================================================
// 预处理 → 前向 → PostProcess（解码 + 坐标还原 + 过滤 + NMS 全在后处理内完成）。
// 模型/任务差异全部由 create_postprocess() 装配的 Decoder + PostProcess 承担，
// 本文件不含任何与具体模型格式相关的解码代码。
InferResult Model::run_pipeline(const cv::Mat& image) {
    // 1. Preprocess
    auto pre = preprocess(image, config_.input_width, config_.input_height);

    // 2. Assemble postprocess（懒初始化，跟随最新的 config_ / task_type）
    if (!postprocess_) {
        postprocess_ = create_postprocess(config_.task_type, config_.model_type,
                                          make_postprocess_params());
        if (!postprocess_) return InferResult{};
    }

    // 解码上下文随config_ 派生（网格尺寸必须用实际输入宽高）
    DecodeContext ctx;
    ctx.num_classes   = config_.num_classes;
    ctx.num_keypoints = config_.num_keypoints;
    ctx.score_thresh  = config_.score_thresh;
    ctx.input_width   = config_.input_width;
    ctx.input_height  = config_.input_height;
    postprocess_->set_context(ctx);

    // 3. Forward
    auto input_names = backend_->get_input_names();
    std::vector<std::vector<int64_t>> input_shapes = {pre.shape()};
    std::vector<cv::Mat> input_data = {pre.blob};

    auto output_names = backend_->get_output_names();
    TensorSet outputs;

    if (!backend_->forward(input_names, input_shapes, input_data,
                           output_names, outputs)) {
        std::cerr << "[" << model_type_name(config_.model_type)
                  << "] Inference failed" << std::endl;
        return InferResult{};
    }

    // 4. PostProcess: decode → restore → filter → NMS
    return postprocess_->forward(outputs, pre.letterbox);
}

InferResult Model::infer(const cv::Mat& image) {
    return run_pipeline(image);
}

DetectResult Model::infer_detect(const cv::Mat& image) {
    auto r = run_pipeline(image);
    if (auto* d = std::get_if<DetectResult>(&r)) return *d;
    return DetectResult{};
}

SegmentResult Model::infer_segment(const cv::Mat& image) {
    auto r = run_pipeline(image);
    if (auto* s = std::get_if<SegmentResult>(&r)) return *s;
    return SegmentResult{};
}

PoseResult Model::infer_pose(const cv::Mat& image) {
    auto r = run_pipeline(image);
    if (auto* p = std::get_if<PoseResult>(&r)) return *p;
    return PoseResult{};
}

OBBResult Model::infer_obb(const cv::Mat& image) {
    auto r = run_pipeline(image);
    if (auto* o = std::get_if<OBBResult>(&r)) return *o;
    return OBBResult{};
}

// ============================================================
// Model Factory
// ============================================================
std::shared_ptr<Model> create_model(ModelType type) {
    return create_model(type, TaskType::Detect);
}

std::shared_ptr<Model> create_model(ModelType type, TaskType task) {
    // 只有 v8 系支持 segment / pose / obb，其余模型仅支持检测
    const bool v8_family = (type == ModelType::YOLOv8 ||
                            type == ModelType::YOLOv11 ||
                            type == ModelType::YOLO26);
    if (!v8_family && task != TaskType::Detect) {
        std::cerr << "[create_model] Task '" << task_type_name(task)
                  << "' only supported for YOLOv8 family" << std::endl;
        return nullptr;
    }
    if (!create_decoder(type)) {
        std::cerr << "[create_model] Unsupported model type: "
                  << model_type_name(type) << std::endl;
        return nullptr;
    }

    // 模型不再派生子类：全部差异由 config_（model_type / task_type / 阈值）表达
    auto model = std::make_shared<Model>();
    model->set_model_task(type, task);
    return model;
}

} // namespace yolo_onnx