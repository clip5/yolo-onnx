#include "yolo_onnx/model.hpp"
#include "yolo_onnx/models/model_v8_obb.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_type> <model.onnx> <image> <output> [options]\n\n"
              << "Model types: v8, v11, v26\n\n"
              << "Options:\n"
              << "  --score=<float>    Score threshold (default: 0.5)\n"
              << "  --nms=<float>      NMS threshold (default: 0.45)\n"
              << "  --size=<int>       Input size (default: 640)\n"
              << "  --classes=<int>    Number of classes (default: 80)\n"
              << "  --backend=<str>    Inference backend (default: onnxruntime)\n"
              << "  --threads=<int>    CPU threads (default: 4)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return -1;
    }

    std::string model_type_str = argv[1];
    std::string model_path     = argv[2];
    std::string image_path     = argv[3];
    std::string out_path       = argv[4];

    float score_thresh = 0.5f;
    float nms_thresh   = 0.45f;
    int   input_size   = 640;
    int   num_classes  = 80;
    std::string backend = "onnxruntime";
    int   num_threads  = 4;

    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--score=") == 0)      score_thresh  = std::stof(arg.substr(8));
        else if (arg.find("--nms=") == 0)   nms_thresh    = std::stof(arg.substr(6));
        else if (arg.find("--size=") == 0)  input_size    = std::stoi(arg.substr(7));
        else if (arg.find("--classes=") == 0) num_classes  = std::stoi(arg.substr(10));
        else if (arg.find("--backend=") == 0) backend      = arg.substr(10);
        else if (arg.find("--threads=") == 0) num_threads  = std::stoi(arg.substr(10));
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    yolo_onnx::ModelType model_type;
    if (model_type_str == "v8")         model_type = yolo_onnx::ModelType::YOLOv8;
    else if (model_type_str == "v11")   model_type = yolo_onnx::ModelType::YOLOv11;
    else if (model_type_str == "v26")   model_type = yolo_onnx::ModelType::YOLO26;
    else {
        std::cerr << "Unknown model type: " << model_type_str << " (obb only supports v8/v11/v26)" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // Create model with OBB task
    auto model = yolo_onnx::create_model(model_type, yolo_onnx::TaskType::OBB);
    if (!model) {
        std::cerr << "Failed to create model" << std::endl;
        return -1;
    }

    // Configure and load
    yolo_onnx::Model::Config config;
    config.model_path    = model_path;
    config.model_type    = model_type;
    config.task_type     = yolo_onnx::TaskType::OBB;
    config.backend_type  = backend;
    config.score_thresh  = score_thresh;
    config.nms_thresh    = nms_thresh;
    config.input_width   = input_size;
    config.input_height  = input_size;
    config.num_classes   = num_classes;
    config.num_threads   = num_threads;

    if (!model->load(config)) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    // Run OBB inference
    auto* obb_model = dynamic_cast<yolo_onnx::ModelV8OBB*>(model.get());
    if (!obb_model) {
        std::cerr << "Model is not an OBB model" << std::endl;
        return -1;
    }
    auto result = obb_model->infer_obb(image);

    std::cout << "Detected " << result.obb_boxes.size() << " oriented objects:" << std::endl;
    for (size_t i = 0; i < result.obb_boxes.size(); i++) {
        const auto& obb = result.obb_boxes[i];
        std::cout << "  #" << i << " label=" << obb.label
                  << " score=" << obb.score
                  << " center=(" << (int)obb.cx << "," << (int)obb.cy << ")"
                  << " size=" << (int)obb.w << "x" << (int)obb.h
                  << " angle=" << obb.angle << " rad"
                  << std::endl;
    }

    // Draw results
    cv::Mat result_img = image.clone();
    cv::RNG rng(0xDEADBEEF);

    for (size_t i = 0; i < result.obb_boxes.size(); i++) {
        const auto& obb = result.obb_boxes[i];

        // Random color
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        // Get the 4 corners of the rotated box
        auto pts = obb.corners();
        std::vector<cv::Point> poly;
        for (const auto& p : pts) {
            poly.push_back(cv::Point((int)p.x, (int)p.y));
        }

        // Draw rotated rectangle
        cv::polylines(result_img, poly, true, color, 2);

        // Draw label
        std::string label = std::to_string(obb.label) + ": " + std::to_string(obb.score).substr(0, 4);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(result_img,
            cv::Point((int)pts[0].x, (int)pts[0].y - label_size.height - 5),
            cv::Point((int)pts[0].x + label_size.width, (int)pts[0].y),
            color, cv::FILLED);
        cv::putText(result_img, label,
            cv::Point((int)pts[0].x, (int)pts[0].y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite(out_path, result_img);
    std::cout << "Result saved to: " << out_path << std::endl;

    return 0;
}