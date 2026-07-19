#include "yolo_onnx/types.hpp"
#include "yolo_onnx/model.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

int test_count = 0;
int pass_count = 0;

#define TEST(name, expr) do { \
    test_count++; \
    bool ok = (expr); \
    if (ok) { pass_count++; } \
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " " << name << std::endl; \
    assert(ok); \
} while(0)

void test_types() {
    std::cout << "\n=== Test: Core Types ===\n" << std::endl;

    // Box
    yolo_onnx::Box box(10, 20, 100, 200, 0.85f, 5);
    TEST("Box constructor", box.x1 == 10 && box.y1 == 20 && box.x2 == 100 && box.y2 == 200);
    TEST("Box width/height", std::abs(box.width() - 90) < 1e-5 && std::abs(box.height() - 180) < 1e-5);
    TEST("Box area", std::abs(box.area() - 16200) < 1e-5);

    // Sigmoid
    TEST("sigmoid(0) = 0.5", std::abs(yolo_onnx::sigmoid(0.0f) - 0.5f) < 1e-5);
    TEST("sigmoid(large) ~ 1", yolo_onnx::sigmoid(100.0f) > 0.999f);
    TEST("sigmoid(-large) ~ 0", yolo_onnx::sigmoid(-100.0f) < 0.001f);

    // IOU
    yolo_onnx::Box a(0, 0, 100, 100, 1.0f, 0);
    yolo_onnx::Box b(50, 50, 150, 150, 1.0f, 0);
    yolo_onnx::Box c(200, 200, 300, 300, 1.0f, 0);
    TEST("IOU overlapping", std::abs(yolo_onnx::iou(a, b) - 2500.0f/17500.0f) < 1e-4);
    TEST("IOU no overlap", yolo_onnx::iou(a, c) < 1e-5);

    // NMS
    std::vector<yolo_onnx::Box> boxes = {
        yolo_onnx::Box(10, 10, 100, 100, 0.9f, 0),
        yolo_onnx::Box(15, 15, 105, 105, 0.8f, 0),
        yolo_onnx::Box(50, 50, 150, 150, 0.7f, 0),
        yolo_onnx::Box(200, 200, 300, 300, 0.95f, 0),
    };
    auto keep = yolo_onnx::nms(boxes, 0.5f);
    TEST("NMS keeps 3 boxes", keep.size() == 3);
    // Box 3 (0.95, no overlap), Box 0 (0.9), Box 2 (0.7, low IOU w/ box0)
    // Box 1 (0.8) is removed by NMS with box 0 (IOU=0.805 > 0.5)

    // Scale box
    yolo_onnx::LetterboxInfo info = {0.5f, 10, 10, 640, 480, 640, 640};
    yolo_onnx::Box scaled(10, 10, 100, 100, 1.0f, 0);
    yolo_onnx::scale_box(scaled, info);
    TEST("Scale box mapping", std::abs(scaled.x1 - 0) < 1e-5 && std::abs(scaled.y1 - 0) < 1e-5);

    // ModelType name
    TEST("ModelType name", std::string(yolo_onnx::model_type_name(yolo_onnx::ModelType::YOLOv8)) == "YOLOv8");
    TEST("ModelType name PPYOLOE", std::string(yolo_onnx::model_type_name(yolo_onnx::ModelType::PPYOLOE)) == "PPYOLOE");
}

void test_model_creation() {
    std::cout << "\n=== Test: Model Factory ===\n" << std::endl;

    auto model_v5 = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv5);
    TEST("Create YOLOv5 model", model_v5 != nullptr);

    auto model_yolox = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOX);
    TEST("Create YOLOX model", model_yolox != nullptr);

    auto model_v8 = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8);
    TEST("Create YOLOv8 model", model_v8 != nullptr);

    auto model_v11 = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv11);
    TEST("Create YOLOv11 model", model_v11 != nullptr);

    auto model_v26 = yolo_onnx::create_model(yolo_onnx::ModelType::YOLO26);
    TEST("Create YOLO26 model", model_v26 != nullptr);

    auto model_ppyoloe = yolo_onnx::create_model(yolo_onnx::ModelType::PPYOLOE);
    TEST("Create PPYOLOE model", model_ppyoloe != nullptr);
}

int main() {
    std::cout << "=== yolo-onnx Unit Tests ===\n" << std::endl;

    test_types();
    test_model_creation();

    std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n" << std::endl;
    return (pass_count == test_count) ? 0 : 1;
}