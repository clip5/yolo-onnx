#include "yolo_onnx/yolo_onnx.hpp"
#include "model_impl.hpp"

#include "core/onnxruntime_backend.hpp"
#include "process/preprocess/preprocess.hpp"
#include "process/postprocess/postprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <utility>

namespace yolo_onnx {

using detail::ModelImpl;

// ============================================================
// Model 生命周期（PIMPL）
// ============================================================
Model::Model()  : impl_(std::make_unique<ModelImpl>()) {}
Model::~Model() = default;

Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

// ============================================================
// 配置
// ============================================================
void Model::set_model_task(ModelType type, TaskType task) {
    config_.model_type = type;
    config_.task_type  = task;
}

const Model::Config& Model::config() const { return config_; }

std::shared_ptr<Backend> Model::backend() const {
    return impl_ ? impl_->backend : nullptr;
}

// ============================================================
// load
// ============================================================
bool Model::load(const Config& config) {
    config_ = config;

    // 1. Create backend
    impl_->backend = create_backend(config.backend_type);
    if (!impl_->backend) {
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

    if (!impl_->backend->load(backend_config)) {
        std::cerr << "[Model] Failed to load model: " << config.model_path << std::endl;
        return false;
    }

    // 3. Override input dimensions from model if available
    auto input_shapes = impl_->backend->get_input_shapes();
    if (!input_shapes.empty() && input_shapes[0].size() >= 4) {
        if (input_shapes[0][2] > 0) config_.input_height = (int)input_shapes[0][2];
        if (input_shapes[0][3] > 0) config_.input_width  = (int)input_shapes[0][3];
    }

    std::cout << "[" << model_type_name(config_.model_type)
              << "] Model loaded: " << config_.model_path
              << " (" << impl_->backend->name() << ")"
              << " input: " << config_.input_width << "x" << config_.input_height
              << std::endl;

    return true;
}

// ============================================================
// 内部流程（实现见 detail::ModelImpl）
// ============================================================
PreProcessResult ModelImpl::preprocess(const cv::Mat& image, ModelType model_type,
                                       int target_w, int target_h) const {
    // 复用独立的 PreProcess 模块（按输入尺寸懒初始化，内部画布/通道缓冲也复用）
    // 注意：预处理差异（YOLOX 的 mean/std、PPYOLOE 的 swap_rb）由 model_type 决定，
    // 不能写死，否则会丢失这些模型的归一化设置。
    if (!preprocessor ||
        preprocessor->params().target_width != target_w ||
        preprocessor->params().target_height != target_h ||
        preprocessor->params().model_type != model_type) {
        auto params = PreProcessParams::for_model(model_type, target_w, target_h);
        preprocessor = std::make_shared<PreProcess>(params);
    }
    return preprocessor->run(image);
}

InferResult ModelImpl::run_pipeline(const cv::Mat& image, const Model::Config& cfg) {
    // 1. Preprocess
    auto pre = preprocess(image, cfg.model_type, cfg.input_width, cfg.input_height);

    // 2. Assemble postprocess（懒初始化，跟随最新的 config_ / task_type）
    if (!postprocess) {
        PostProcessParams pp;
        pp.score_thresh = cfg.score_thresh;
        pp.nms_thresh   = cfg.nms_thresh;
        postprocess = create_postprocess(cfg.task_type, cfg.model_type, pp);
        if (!postprocess) return InferResult{};
    }

    // 解码上下文随 config_ 派生（网格尺寸必须用实际输入宽高，不能硬编码）
    DecodeContext ctx;
    ctx.num_classes   = cfg.num_classes;
    ctx.num_keypoints = cfg.num_keypoints;
    ctx.score_thresh  = cfg.score_thresh;
    ctx.input_width   = cfg.input_width;
    ctx.input_height  = cfg.input_height;
    postprocess->set_context(ctx);

    // 3. Forward
    auto input_names  = backend->get_input_names();
    std::vector<std::vector<int64_t>> input_shapes = {pre.shape()};
    std::vector<cv::Mat> input_data = {pre.blob};

    auto output_names = backend->get_output_names();
    TensorSet outputs;

    if (!backend->forward(input_names, input_shapes, input_data,
                          output_names, outputs)) {
        std::cerr << "[" << model_type_name(cfg.model_type)
                  << "] Inference failed" << std::endl;
        return InferResult{};
    }

    // 4. PostProcess: decode → restore → filter → NMS
    return postprocess->forward(outputs, pre.letterbox);
}

// ============================================================
// 推理入口
// ============================================================
InferResult Model::infer(const cv::Mat& image) {
    return impl_->run_pipeline(image, config_);
}

DetectResult Model::infer_detect(const cv::Mat& image) {
    auto r = impl_->run_pipeline(image, config_);
    if (auto* d = std::get_if<DetectResult>(&r)) return *d;
    return DetectResult{};
}

SegmentResult Model::infer_segment(const cv::Mat& image) {
    auto r = impl_->run_pipeline(image, config_);
    if (auto* s = std::get_if<SegmentResult>(&r)) return *s;
    return SegmentResult{};
}

PoseResult Model::infer_pose(const cv::Mat& image) {
    auto r = impl_->run_pipeline(image, config_);
    if (auto* p = std::get_if<PoseResult>(&r)) return *p;
    return PoseResult{};
}

OBBResult Model::infer_obb(const cv::Mat& image) {
    auto r = impl_->run_pipeline(image, config_);
    if (auto* o = std::get_if<OBBResult>(&r)) return *o;
    return OBBResult{};
}

SemResult Model::infer_sem(const cv::Mat& image) {
    auto r = impl_->run_pipeline(image, config_);
    if (auto* s = std::get_if<SemResult>(&r)) return *s;
    return SemResult{};
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

    // 模型不再派生子类：全部差异由 Config（model_type / task_type / 阈值）表达
    auto model = std::make_shared<Model>();
    model->set_model_task(type, task);
    return model;
}

} // namespace yolo_onnx