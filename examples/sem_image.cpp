#include "yolo_onnx/yolo_onnx.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_type> <model.onnx> <image> <output> [options]\n\n"
              << "Model types: v8, v11, v26\n\n"
              << "Options:\n"
              << "  --classes=<int>    Number of classes (default: 19, cityscapes)\n"
              << "  --alpha=<float>    Overlay opacity 0..1 (default: 0.55)\n"
              << "  --legend           Draw the class legend on the image\n"
              << "  --backend=<str>    Inference backend (default: onnxruntime)\n"
              << "  --ep=<str>         Execution provider (default: cpu)\n"
              << "  --device=<int>     Device id (default: 0)\n"
              << "  --threads=<int>    CPU threads (default: 4)\n"
              << std::endl;
}

namespace {

// Ultralytics 默认调色板（与 ultralytics.utils.plotting.Colors 一致），
// 取前 kClasses 个。语义分割本身没有实例 id，类别 id 就是配色下标，
// 因此两边的配色天然可对齐，方便肉眼逐类核对。
const uint8_t kPalette[20][3] = {
    {  4,  42, 255}, { 11, 219, 235}, {243, 243, 243}, {  0, 223, 183}, { 17,  31, 104},
    {255, 111, 221}, {255,  68,  79}, {204, 237,   0}, {  0, 243,  68}, {189,   0, 255},
    {  0, 180, 255}, {221,   0, 186}, {  0, 255, 255}, { 38, 192,   0}, {  1, 255, 179},
    {125,  36, 255}, {123,   0, 104}, {255,  27, 108}, {252, 109,  47}, {162, 255,  11},
};

// cityscapes.yaml 的 19 类名（yolo26s-sem.onnx 的 metadata.names）
const char* kCityscapes[19] = {
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return -1;
    }

    const std::string model_type_str = argv[1];
    const std::string model_path     = argv[2];
    const std::string image_path     = argv[3];
    const std::string out_path       = argv[4];

    int   num_classes = 19;
    float alpha       = 0.55f;
    bool  draw_legend = false;
    std::string backend   = "onnxruntime";
    std::string ep        = "cpu";
    int   num_threads    = 4;
    int   device_id      = 0;

    for (int i = 5; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg.rfind("--classes=", 0) == 0)     num_classes = std::stoi(arg.substr(10));
        else if (arg.rfind("--alpha=", 0) == 0)  alpha       = std::stof(arg.substr(8));
        else if (arg == "--legend")              draw_legend = true;
        else if (arg.rfind("--backend=", 0) == 0) backend     = arg.substr(10);
        else if (arg.rfind("--ep=", 0) == 0)     ep          = arg.substr(5);
        else if (arg.rfind("--device=", 0) == 0) device_id   = std::stoi(arg.substr(9));
        else if (arg.rfind("--threads=", 0) == 0) num_threads = std::stoi(arg.substr(10));
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }
    alpha = std::max(0.0f, std::min(1.0f, alpha));

    yolo_onnx::ModelType model_type;
    if (model_type_str == "v8")       model_type = yolo_onnx::ModelType::YOLOv8;
    else if (model_type_str == "v11") model_type = yolo_onnx::ModelType::YOLOv11;
    else if (model_type_str == "v26") model_type = yolo_onnx::ModelType::YOLO26;
    else {
        std::cerr << "Unknown model type: " << model_type_str
                  << " (sem only supports v8/v11/v26)" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    auto model = yolo_onnx::create_model(model_type, yolo_onnx::TaskType::Sem);
    if (!model) {
        std::cerr << "Failed to create model" << std::endl;
        return -1;
    }

    yolo_onnx::Model::Config config;
    config.model_path   = model_path;
    config.model_type   = model_type;
    config.task_type    = yolo_onnx::TaskType::Sem;
    config.backend_type = backend;
    config.input_width  = -1;   // 让 load() 读模型自身的输入尺寸（1024）
    config.input_height = -1;
    config.num_classes  = num_classes;
    config.num_threads  = num_threads;
    config.custom_config = "ep=" + ep + ";--device=" + std::to_string(device_id);

    if (!model->load(config)) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    yolo_onnx::InferResult result = model->infer(image);
    auto* sem = std::get_if<yolo_onnx::SemResult>(&result);
    if (!sem) {
        std::cerr << "Expected sem result" << std::endl;
        return -1;
    }

    const auto& mask = sem->mask;
    if (mask.width != image.cols || mask.height != image.rows) {
        std::cerr << "Warning: mask size " << mask.width << "x" << mask.height
                  << " != image size " << image.cols << "x" << image.rows << std::endl;
    }

    // 统计每类像素数 —— 这是核对精度的第一手数字（比肉眼看图可靠得多）
    std::map<int, size_t> hist;
    for (size_t i = 0; i < mask.data.size(); i++) {
        hist[(int)mask.data[i]]++;
    }
    const size_t total_px = mask.data.empty() ? 1 : mask.data.size();
    std::cout << "Semantic segmentation: " << mask.width << "x" << mask.height
              << ", " << hist.size() << " classes present" << std::endl;
    for (const auto& kv : hist) {
        const int cls = kv.first;
        const char* name = (cls >= 0 && cls < num_classes &&
                            cls < (int)(sizeof(kCityscapes) / sizeof(kCityscapes[0])))
                           ? kCityscapes[cls] : "?";
        std::cout << "  class " << cls << " (" << name << "): " << kv.second
                  << " px (" << std::fixed
                  << 100.0 * kv.second / (double)total_px << "%)" << std::endl;
    }

    // 上色：按类别 id 取调色板，与原图按 alpha 混合
    // kPalette 是 RGB 序（与 ultralytics Colors 一致），cv::Mat 是 BGR 序，
    // 写入时必须翻转 R/B——原样写入会把路面画成红色、人行道画成黄色。
    cv::Mat color_img(mask.height, mask.width, CV_8UC3);
    for (int y = 0; y < mask.height; y++) {
        const float* row = mask.data.data() + (size_t)y * mask.width;
        cv::Vec3b* dst = color_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.width; x++) {
            int cls = (int)row[x];
            if (cls < 0 || cls >= 20) cls = 0;   // 越界 id 归到 0，避免越界读
            dst[x] = cv::Vec3b(kPalette[cls][2], kPalette[cls][1], kPalette[cls][0]);
        }
    }

    cv::Mat canvas;
    cv::resize(color_img, canvas, image.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat result_img;
    cv::addWeighted(image, 1.0 - alpha, canvas, alpha, 0.0, result_img);

    if (draw_legend) {
        // 图例只列出图中实际出现的类别，按像素数从多到少
        std::vector<std::pair<int, size_t>> items(hist.begin(), hist.end());
        std::sort(items.begin(), items.end(),
                  [](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
                      return a.second > b.second;
                  });
        const int line_h = 22;
        const int max_lines = (int)items.size();
        const int box_h = 8 + line_h * max_lines;
        const int box_w = 210;
        cv::rectangle(result_img,
                      cv::Point(10, 10),
                      cv::Point(10 + box_w, 10 + box_h),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        for (int i = 0; i < max_lines; i++) {
            const int cls = items[i].first;
            const char* name = (cls >= 0 && cls < num_classes &&
                                cls < (int)(sizeof(kCityscapes) / sizeof(kCityscapes[0])))
                               ? kCityscapes[cls] : "?";
            const int y = 26 + i * line_h;
            const int idx = (cls >= 0 && cls < 20) ? cls : 0;
            // 图例色块同样要 RGB→BGR 翻转（cv::Scalar 也是 BGR 序）
            cv::rectangle(result_img, cv::Point(18, y - 12), cv::Point(34, y + 4),
                          cv::Scalar(kPalette[idx][2], kPalette[idx][1], kPalette[idx][0]),
                          cv::FILLED);
            std::string text = std::string(name) + "  " +
                               std::to_string((int)items[i].second) + "px";
            cv::putText(result_img, text, cv::Point(42, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

    cv::imwrite(out_path, result_img);
    std::cout << "Result saved to: " << out_path << std::endl;
    return 0;
}