#include "yolo_onnx/yolo_onnx.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_type> <model.onnx> <image> <output> [options]\n\n"
              << "Model types: v8, v11, v26\n\n"
              << "Options:\n"
              << "  --score=<float>    Score threshold (default: 0.5)\n"
              << "  --nms=<float>      NMS threshold (default: 0.45)\n"
              << "  --size=<int>       Input size (default: 640)\n"
              << "  --classes=<int>    Number of classes (default: 1)\n"
              << "  --keypoints=<int>  Number of keypoints (default: 17)\n"
              << "  --backend=<str>    Inference backend (default: onnxruntime)\n"
              << "  --threads=<int>    CPU threads (default: 4)\n"
              << std::endl;
}

// COCO keypoint skeleton connections (pairs of keypoint indices)
static const std::vector<std::pair<int, int>> skeleton = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
    {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11},
    {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}
};

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
    int   num_classes  = 1;   // pose usually has 1 class (person)
    int   num_keypoints = 17; // COCO keypoints
    std::string backend = "onnxruntime";
    int   num_threads  = 4;

    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--score=") == 0)      score_thresh   = std::stof(arg.substr(8));
        else if (arg.find("--nms=") == 0)   nms_thresh     = std::stof(arg.substr(6));
        else if (arg.find("--size=") == 0)  input_size     = std::stoi(arg.substr(7));
        else if (arg.find("--classes=") == 0) num_classes   = std::stoi(arg.substr(10));
        else if (arg.find("--keypoints=") == 0) num_keypoints = std::stoi(arg.substr(12));
        else if (arg.find("--backend=") == 0) backend       = arg.substr(10);
        else if (arg.find("--threads=") == 0) num_threads   = std::stoi(arg.substr(10));
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
        std::cerr << "Unknown model type: " << model_type_str << " (pose only supports v8/v11/v26)" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // Create model with pose task
    auto model = yolo_onnx::create_model(model_type, yolo_onnx::TaskType::Pose);
    if (!model) {
        std::cerr << "Failed to create model" << std::endl;
        return -1;
    }

    // Configure and load
    yolo_onnx::Model::Config config;
    config.model_path    = model_path;
    config.model_type    = model_type;
    config.task_type     = yolo_onnx::TaskType::Pose;
    config.backend_type  = backend;
    config.score_thresh  = score_thresh;
    config.nms_thresh    = nms_thresh;
    config.input_width   = input_size;
    config.input_height  = input_size;
    config.num_classes   = num_classes;
    config.num_keypoints = num_keypoints;
    config.num_threads   = num_threads;

    if (!model->load(config)) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    // Run unified inference — no dynamic_cast needed
    yolo_onnx::InferResult result = model->infer(image);
    auto* pose = std::get_if<yolo_onnx::PoseResult>(&result);
    if (!pose) {
        std::cerr << "Expected pose result" << std::endl;
        return -1;
    }

    std::cout << "Detected " << pose->boxes.size() << " poses:" << std::endl;
    for (size_t i = 0; i < pose->boxes.size(); i++) {
        const auto& box = pose->boxes[i];
        std::cout << "  #" << i << " label=" << box.label
                  << " score=" << box.score
                  << " rect=[" << (int)box.x1 << "," << (int)box.y1
                  << "," << (int)box.x2 << "," << (int)box.y2 << "]"
                  << " keypoints=" << pose->keypoints[i].size()
                  << std::endl;
    }

    // Draw results
    cv::Mat result_img = image.clone();

    for (size_t i = 0; i < pose->boxes.size(); i++) {
        const auto& box = pose->boxes[i];
        const auto& kpts = pose->keypoints[i];

        // Draw box
        cv::rectangle(result_img,
            cv::Point((int)box.x1, (int)box.y1),
            cv::Point((int)box.x2, (int)box.y2),
            cv::Scalar(0, 255, 0), 2);

        // Draw skeleton (keypoint connections)
        for (const auto& [i1, i2] : skeleton) {
            int idx1 = i1 - 1;  // convert to 0-based
            int idx2 = i2 - 1;
            if (idx1 >= (int)kpts.size() || idx2 >= (int)kpts.size()) continue;

            if (kpts[idx1].visibility > 0.5f && kpts[idx2].visibility > 0.5f) {
                cv::line(result_img,
                    cv::Point((int)kpts[idx1].x, (int)kpts[idx1].y),
                    cv::Point((int)kpts[idx2].x, (int)kpts[idx2].y),
                    cv::Scalar(0, 255, 0), 2);
            }
        }

        // Draw keypoints
        for (const auto& kp : kpts) {
            if (kp.visibility > 0.5f) {
                cv::circle(result_img,
                    cv::Point((int)kp.x, (int)kp.y),
                    3, cv::Scalar(0, 0, 255), -1);
            }
        }

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