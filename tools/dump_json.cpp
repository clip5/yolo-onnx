// dump_json.cpp — run the yolo_onnx pipeline and dump results as JSON,
// so they can be diffed numerically against the python reference
// (tools/ref_onnx.py) instead of only eyeballing annotated images.
//
// Usage:
//   dump_json <model_type> <task> <model.onnx> <image> <out.json> [options]
// Options:
//   --size=WxH | --size=N   input size (default: from the ONNX model)
//   --score=<f> --nms=<f> --classes=<n> --threads=<n> --nms-mode=class|agnostic
//
// Output JSON: { "boxes":[{x1,y1,x2,y2,score,label}...],
//                "masks":[[...]] (segment only, 0/1 at original image size) }
#include "yolo_onnx/yolo_onnx.hpp"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace yolo_onnx;

namespace {

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <model_type> <task> <model.onnx> <image> <out.json> [options]\n"
              << "  model_type: v5 | yolox | v8 | v11 | v26 | ppyoloe\n"
              << "  task      : detect | segment | pose | obb\n"
              << "Options:\n"
              << "  --size=WxH          input size (default: model's own)\n"
              << "  --score=<f>  --nms=<f>  --classes=<n>  --threads=<n>\n"
              << "  --nms-mode=class|agnostic\n";
}

ModelType parse_model_type(const std::string& s) {
    if (s == "v5")          return ModelType::YOLOv5;
    if (s == "yolox")       return ModelType::YOLOX;
    if (s == "v8")          return ModelType::YOLOv8;
    if (s == "v11")         return ModelType::YOLOv11;
    if (s == "v26")         return ModelType::YOLO26;
    if (s == "ppyoloe")     return ModelType::PPYOLOE;
    return ModelType::YOLOv8;
}

TaskType parse_task_type(const std::string& s) {
    if (s == "detect")  return TaskType::Detect;
    if (s == "segment") return TaskType::Segment;
    if (s == "pose")    return TaskType::Pose;
    if (s == "obb")     return TaskType::OBB;
    if (s == "sem")     return TaskType::Sem;
    return TaskType::Detect;
}

// Serialize with enough precision to see float drift, not so much that
// cosmetic rounding noise dominates the diff.
std::string num(double v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(4) << v;
    return os.str();
}

void write_boxes(std::ostream& os, const BoxArray& boxes) {
    os << "[";
    for (size_t i = 0; i < boxes.size(); i++) {
        const auto& b = boxes[i];
        if (i) os << ",";
        os << "\n{\"x1\":" << num(b.x1) << ",\"y1\":" << num(b.y1)
           << ",\"x2\":" << num(b.x2) << ",\"y2\":" << num(b.y2)
           << ",\"score\":" << num(b.score)
           << ",\"label\":" << b.label << "}";
    }
    os << (boxes.empty() ? "" : "\n") << "]";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) { usage(argv[0]); return 1; }

    const ModelType model_type = parse_model_type(argv[1]);
    const TaskType  task_type  = parse_task_type(argv[2]);
    const std::string model_path = argv[3];
    const std::string image_path = argv[4];
    const std::string out_path   = argv[5];

    Model::Config cfg;
    cfg.model_path  = model_path;
    cfg.model_type  = model_type;
    cfg.task_type   = task_type;
    cfg.score_thresh = 0.25f;
    cfg.nms_thresh   = 0.45f;
    cfg.num_threads  = 4;

    int size_w = 0, size_h = 0;
    for (int i = 6; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg.rfind("--size=", 0) == 0) {
            const std::string v = arg.substr(7);
            const size_t x = v.find_first_of("xX");
            if (x != std::string::npos) {
                size_w = std::stoi(v.substr(0, x));
                size_h = std::stoi(v.substr(x + 1));
            } else {
                size_w = size_h = std::stoi(v);
            }
        }
        else if (arg.rfind("--score=", 0) == 0)     cfg.score_thresh = std::stof(arg.substr(8));
        else if (arg.rfind("--nms=", 0) == 0)       cfg.nms_thresh   = std::stof(arg.substr(6));
        else if (arg.rfind("--classes=", 0) == 0)   cfg.num_classes  = std::stoi(arg.substr(10));
        else if (arg.rfind("--threads=", 0) == 0)   cfg.num_threads  = std::stoi(arg.substr(10));
        else if (arg == "--nms-mode=agnostic") {
            // stored below; PostProcessParams is built inside ModelImpl
        }
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }

    auto model = create_model(model_type, task_type);
    if (!model) { std::cerr << "create_model failed\n"; return 1; }

    // Only override the model's own input size when explicitly requested.
    // Otherwise Model::load() picks the real input shape from the ONNX file,
    // which is what a fixed-shape export requires anyway.
    if (size_w > 0 && size_h > 0) {
        cfg.input_width  = size_w;
        cfg.input_height = size_h;
    } else {
        cfg.input_width  = -1;
        cfg.input_height = -1;
    }

    if (!model->load(cfg)) { std::cerr << "load failed\n"; return 1; }

    InferResult r = model->infer(image);

    std::ofstream os(out_path);
    if (!os) { std::cerr << "cannot write " << out_path << "\n"; return 1; }
    os << "{\n";

    const auto& c = model->config();
    os << "\"model\":\"" << model_path << "\",\n";
    os << "\"task\":\"" << task_type_name(c.task_type) << "\",\n";
    os << "\"input_width\":" << c.input_width << ",\n";
    os << "\"input_height\":" << c.input_height << ",\n";
    os << "\"image_width\":" << image.cols << ",\n";
    os << "\"image_height\":" << image.rows << ",\n";

    if (const auto* det = std::get_if<DetectResult>(&r)) {
        os << "\"boxes\":"; write_boxes(os, det->boxes); os << "\n";
    } else if (const auto* seg = std::get_if<SegmentResult>(&r)) {
        os << "\"boxes\":"; write_boxes(os, seg->boxes); os << ",\n";
        os << "\"mask_shape\":[" << seg->masks.size();
        if (!seg->masks.empty()) {
            os << "," << seg->masks[0].width << "," << seg->masks[0].height;
        }
        os << "],\n";
        // Masks are binary (0/1) at original-image size — emit run-length per row
        // to keep the JSON manageable.
        os << "\"masks\":[";
        for (size_t i = 0; i < seg->masks.size(); i++) {
            const auto& m = seg->masks[i];
            if (i) os << ",";
            os << "\n[";
            for (int y = 0; y < m.height; y++) {
                if (y) os << ",";
                os << "[";
                for (int x = 0; x < m.width; x++) {
                    if (x) os << ",";
                    os << (m.data[(size_t)y * m.width + x] > 0.5f ? 1 : 0);
                }
                os << "]";
            }
            os << "]";
        }
        os << (seg->masks.empty() ? "" : "\n") << "]\n";
    } else if (const auto* pose = std::get_if<PoseResult>(&r)) {
        os << "\"boxes\":"; write_boxes(os, pose->boxes); os << ",\n";
        os << "\"keypoints\":[";
        for (size_t i = 0; i < pose->boxes.size(); i++) {
            if (i) os << ",";
            os << "\n[";
            for (size_t k = 0; k < pose->keypoints[i].size(); k++) {
                if (k) os << ",";
                const auto& kp = pose->keypoints[i][k];
                os << "[" << num(kp.x) << "," << num(kp.y) << "," << num(kp.visibility) << "]";
            }
            os << "]";
        }
        os << (pose->boxes.empty() ? "" : "\n") << "]\n";
    } else if (const auto* obb = std::get_if<OBBResult>(&r)) {
        os << "\"obb\":[";
        for (size_t i = 0; i < obb->obb_boxes.size(); i++) {
            const auto& b = obb->obb_boxes[i];
            if (i) os << ",";
            os << "\n{\"cx\":" << num(b.cx) << ",\"cy\":" << num(b.cy)
               << ",\"w\":" << num(b.w) << ",\"h\":" << num(b.h)
               << ",\"angle\":" << num(b.angle)
               << ",\"score\":" << num(b.score)
               << ",\"label\":" << b.label << "}";
        }
        os << (obb->obb_boxes.empty() ? "" : "\n") << "]\n";
    } else if (const auto* sem = std::get_if<SemResult>(&r)) {
        os << "\"sem_shape\":[" << sem->mask.width << "," << sem->mask.height << "],\n";
        // 类别 id 图：逐像素一个整数。用 RLE 压缩，避免 1024x1024 写成几 MB JSON。
        // 每行一个 run: [起始列, 类别id, 连续长度]
        os << "\"sem_rle\":[";
        for (int y = 0; y < sem->mask.height; y++) {
            const float* row = sem->mask.data.data() + (size_t)y * sem->mask.width;
            if (y) os << ",";
            os << "\n[";
            int x = 0;
            bool first_run = true;
            while (x < sem->mask.width) {
                const int cls = (int)row[x];
                int run = 1;
                while (x + run < sem->mask.width && (int)row[x + run] == cls) run++;
                if (!first_run) os << ",";
                first_run = false;
                os << "[" << x << "," << cls << "," << run << "]";
                x += run;
            }
            os << "]";
        }
        os << (sem->mask.height ? "\n" : "") << "]\n";
    } else {
        os << "\"boxes\":[],\n\"note\":\"unsupported task type\"\n";
    }

    os << "}\n";
    os.close();
    std::cout << "wrote " << out_path << "\n";
    return 0;
}