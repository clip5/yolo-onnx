#include "yolo_onnx/yolo_onnx.hpp"
#include "core/onnxruntime_backend.hpp"
#include "process/preprocess/preprocess.hpp"
#include "models/detect/v5.hpp"
#include "models/detect/yolox.hpp"
#include "models/detect/v8.hpp"
#include "models/detect/ppyoloe.hpp"
#include "models/segment/v8.hpp"
#include "models/pose/v8.hpp"
#include "models/obb/v8.hpp"
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
// Model Factory (detection — backwards compatible)
// ============================================================
std::shared_ptr<Model> create_model(ModelType type) {
    return create_model(type, TaskType::Detect);
}

// ============================================================
// Model Factory (task-aware)
// ============================================================
std::shared_ptr<Model> create_model(ModelType type, TaskType task) {
    // For non-YOLOv8 model types, only detection is supported
    if (type != ModelType::YOLOv8 && type != ModelType::YOLOv11 && type != ModelType::YOLO26) {
        if (task != TaskType::Detect) {
            std::cerr << "[create_model] Task '" << task_type_name(task)
                      << "' only supported for YOLOv8 family" << std::endl;
            return nullptr;
        }
        switch (type) {
            case ModelType::YOLOv5:   return std::make_shared<ModelV5>();
            case ModelType::YOLOX:    return std::make_shared<ModelYOLOX>();
            case ModelType::PPYOLOE:  return std::make_shared<ModelPPYOLOE>();
            default: break;
        }
    }

    // YOLOv8 family with task dispatch
    switch (task) {
        case TaskType::Detect:  return std::make_shared<ModelV8>();
        case TaskType::Segment: return std::make_shared<ModelV8Segment>();
        case TaskType::Pose:    return std::make_shared<ModelV8Pose>();
        case TaskType::OBB:     return std::make_shared<ModelV8OBB>();
        default:
            std::cerr << "[create_model] Unknown task type" << std::endl;
            return nullptr;
    }
}

} // namespace yolo_onnx