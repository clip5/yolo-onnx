#include "yolo_onnx/model.hpp"
#include "yolo_onnx/backends/onnxruntime_backend.hpp"
#include "yolo_onnx/models/model_v5.hpp"
#include "yolo_onnx/models/model_yolox.hpp"
#include "yolo_onnx/models/model_v8.hpp"
#include "yolo_onnx/models/model_ppyoloe.hpp"
#include "yolo_onnx/models/model_v8_segment.hpp"
#include "yolo_onnx/models/model_v8_pose.hpp"
#include "yolo_onnx/models/model_v8_obb.hpp"
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
    PreProcessResult result;

    int img_w = image.cols;
    int img_h = image.rows;

    // 1. Letterbox resize (maintain aspect ratio with padding)
    float scale = std::min((float)target_w / img_w, (float)target_h / img_h);
    int new_w = (int)(img_w * scale);
    int new_h = (int)(img_h * scale);
    int pad_left = (target_w - new_w) / 2;
    int pad_top  = (target_h - new_h) / 2;

    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    // Pad to target size
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_left, pad_top, new_w, new_h)));

    result.letterbox = {
        scale, pad_left, pad_top,
        img_w, img_h, target_w, target_h
    };

    // 2. Convert to NCHW float blob (BGR → RGB, normalize to [0,1])
    result.blob.resize(3 * target_h * target_w);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < target_h; h++) {
            for (int w = 0; w < target_w; w++) {
                // BGR → RGB channel swap
                int src_c = (config_.model_type == ModelType::PPYOLOE) ? c : (2 - c);
                result.blob[c * target_h * target_w + h * target_w + w] =
                    canvas.at<cv::Vec3b>(h, w)[src_c] / 255.0f;
            }
        }
    }

    return result;
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