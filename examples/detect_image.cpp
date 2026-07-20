#include "yolo_onnx/yolo_onnx.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_type> <model.onnx> <image> <output> [options]\n\n"
              << "Model types:\n"
              << "  v5      YOLOv5\n"
              << "  yolox   YOLOX\n"
              << "  v8      YOLOv8\n"
              << "  v11     YOLOv11\n"
              << "  v26     YOLO26\n"
              << "  ppyoloe PPYOLOE\n\n"
              << "Options:\n"
              << "  --score=<float>    Score threshold (default: 0.5)\n"
              << "  --nms=<float>      NMS threshold (default: 0.45)\n"
              << "  --size=<int>       Input size (default: 640)\n"
              << "  --classes=<int>    Number of classes (default: 80)\n"
              << "  --backend=<str>    Inference backend (default: onnxruntime)\n"
              << "                     Supported: onnxruntime, tensorrt, cann, rknn\n"
              << "  --device=<int>     Device ID for GPU/NPU (default: 0)\n"
              << "  --fp16             Enable FP16 inference (TensorRT)\n"
              << "  --threads=<int>    CPU threads (default: 4)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return -1;
    }

    // Parse args
    std::string model_type_str = argv[1];
    std::string model_path     = argv[2];
    std::string image_path     = argv[3];
    std::string out_path       = argv[4];

    // Default params
    float score_thresh = 0.5f;
    float nms_thresh   = 0.45f;
    int   input_size   = 640;
    int   num_classes  = 80;
    std::string backend = "onnxruntime";
    int   num_threads  = 4;
    int   device_id    = 0;
    bool  enable_fp16  = false;

    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--score=") == 0)      score_thresh  = std::stof(arg.substr(8));
        else if (arg.find("--nms=") == 0)   nms_thresh    = std::stof(arg.substr(6));
        else if (arg.find("--size=") == 0)  input_size    = std::stoi(arg.substr(7));
        else if (arg.find("--classes=") == 0) num_classes  = std::stoi(arg.substr(10));
        else if (arg.find("--backend=") == 0) backend      = arg.substr(10);
        else if (arg.find("--device=") == 0) device_id    = std::stoi(arg.substr(9));
        else if (arg.find("--fp16") == 0)    enable_fp16  = true;
        else if (arg.find("--threads=") == 0) num_threads  = std::stoi(arg.substr(10));
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    // Map model type string to enum
    yolo_onnx::ModelType model_type;
    if (model_type_str == "v5")         model_type = yolo_onnx::ModelType::YOLOv5;
    else if (model_type_str == "yolox") model_type = yolo_onnx::ModelType::YOLOX;
    else if (model_type_str == "v8")    model_type = yolo_onnx::ModelType::YOLOv8;
    else if (model_type_str == "v11")   model_type = yolo_onnx::ModelType::YOLOv11;
    else if (model_type_str == "v26")   model_type = yolo_onnx::ModelType::YOLO26;
    else if (model_type_str == "ppyoloe") model_type = yolo_onnx::ModelType::PPYOLOE;
    else {
        std::cerr << "Unknown model type: " << model_type_str << std::endl;
        print_usage(argv[0]);
        return -1;
    }

    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // Create model
    auto model = yolo_onnx::create_model(model_type);
    if (!model) {
        std::cerr << "Failed to create model" << std::endl;
        return -1;
    }

    // Configure and load
    yolo_onnx::Model::Config config;
    config.model_path   = model_path;
    config.model_type   = model_type;
    config.backend_type = backend;
    config.score_thresh = score_thresh;
    config.nms_thresh   = nms_thresh;
    config.input_width  = input_size;
    config.input_height = input_size;
    config.num_classes  = num_classes;
    config.num_threads  = num_threads;

    // Backend-specific config via custom_config
    config.custom_config = "--device=" + std::to_string(device_id);
    if (enable_fp16) {
        config.custom_config += ";--fp16";
    }

    if (!model->load(config)) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    // Run inference
    yolo_onnx::InferResult result = model->infer(image);
    auto* det = std::get_if<yolo_onnx::DetectResult>(&result);
    if (!det) {
        std::cerr << "Unexpected result type" << std::endl;
        return -1;
    }
    const auto& boxes = det->boxes;

    std::cout << "Detected " << boxes.size() << " objects:" << std::endl;
    for (const auto& box : boxes) {
        std::cout << "  label=" << box.label
                  << " score=" << box.score
                  << " rect=[" << (int)box.x1 << "," << (int)box.y1
                  << "," << (int)box.x2 << "," << (int)box.y2 << "]"
                  << std::endl;
    }

    // Draw results
    cv::Mat result_img = image.clone();
    for (const auto& box : boxes) {
        cv::rectangle(result_img,
            cv::Point((int)box.x1, (int)box.y1),
            cv::Point((int)box.x2, (int)box.y2),
            cv::Scalar(0, 255, 0), 2);

        std::string label = std::to_string(box.label) + ": " + std::to_string(box.score).substr(0, 4);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(result_img,
            cv::Point((int)box.x1, (int)box.y1 - label_size.height - 5),
            cv::Point((int)box.x1 + label_size.width, (int)box.y1),
            cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(result_img, label,
            cv::Point((int)box.x1, (int)box.y1 - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite(out_path, result_img);
    std::cout << "Result saved to: " << out_path << std::endl;

    return 0;
}