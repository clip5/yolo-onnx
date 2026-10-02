#include "yolo_onnx/yolo_onnx.hpp"
#include "process/postprocess/postprocess_core.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <variant>

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

void test_task_factory() {
    std::cout << "\n=== Test: Task-Aware Factory ===\n" << std::endl;

    // Detect
    auto det = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8, yolo_onnx::TaskType::Detect);
    TEST("Create YOLOv8 detect", det != nullptr);

    // Segment
    auto seg = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8, yolo_onnx::TaskType::Segment);
    TEST("Create YOLOv8 segment", seg != nullptr);

    // Pose
    auto pose = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8, yolo_onnx::TaskType::Pose);
    TEST("Create YOLOv8 pose", pose != nullptr);

    // OBB
    auto obb = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8, yolo_onnx::TaskType::OBB);
    TEST("Create YOLOv8 obb", obb != nullptr);

    // Non-YOLOv8 models should only support detect
    auto v5_seg = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv5, yolo_onnx::TaskType::Segment);
    TEST("YOLOv5 segment returns nullptr", v5_seg == nullptr);
}

void test_infer_result_types() {
    std::cout << "\n=== Test: InferResult Variant ===\n" << std::endl;

    // DetectResult
    yolo_onnx::DetectResult det;
    det.boxes.push_back(yolo_onnx::Box(0, 0, 10, 10, 0.9f, 1));
    yolo_onnx::InferResult ir = det;
    TEST("InferResult holds DetectResult", std::holds_alternative<yolo_onnx::DetectResult>(ir));
    TEST("DetectResult boxes count", std::get<yolo_onnx::DetectResult>(ir).boxes.size() == 1);

    // SegmentResult
    yolo_onnx::SegmentResult seg;
    seg.boxes.push_back(yolo_onnx::Box(0, 0, 20, 20, 0.8f, 2));
    seg.masks.emplace_back(640, 480);
    ir = seg;
    TEST("InferResult holds SegmentResult", std::holds_alternative<yolo_onnx::SegmentResult>(ir));
    auto* seg_ptr = std::get_if<yolo_onnx::SegmentResult>(&ir);
    TEST("SegmentResult has boxes", seg_ptr != nullptr && seg_ptr->boxes.size() == 1);
    TEST("SegmentResult has masks", seg_ptr != nullptr && seg_ptr->masks.size() == 1);

    // PoseResult
    yolo_onnx::PoseResult pose;
    pose.boxes.push_back(yolo_onnx::Box(30, 30, 60, 60, 0.7f, 0));
    pose.keypoints.push_back({yolo_onnx::Keypoint{10, 20, 2.0f}, yolo_onnx::Keypoint{30, 40, 2.0f}});
    ir = pose;
    TEST("InferResult holds PoseResult", std::holds_alternative<yolo_onnx::PoseResult>(ir));
    auto* pose_ptr = std::get_if<yolo_onnx::PoseResult>(&ir);
    TEST("PoseResult has keypoints", pose_ptr != nullptr && pose_ptr->keypoints.size() == 1);
    TEST("Pose keypoints count", pose_ptr != nullptr && pose_ptr->keypoints[0].size() == 2);

    // OBBResult
    yolo_onnx::OBBResult obb;
    obb.obb_boxes.push_back(yolo_onnx::OBBBox(50, 50, 100, 50, 0.5f, 0.9f, 3));
    ir = obb;
    TEST("InferResult holds OBBResult", std::holds_alternative<yolo_onnx::OBBResult>(ir));
    auto* obb_ptr = std::get_if<yolo_onnx::OBBResult>(&ir);
    TEST("OBBResult has boxes", obb_ptr != nullptr && obb_ptr->obb_boxes.size() == 1);
    TEST("OBBBox angle", obb_ptr != nullptr && std::abs(obb_ptr->obb_boxes[0].angle - 0.5f) < 1e-5);
}

void test_obb_result() {
    std::cout << "\n=== Test: OBB Result ===\n" << std::endl;

    // OBBBox corners
    yolo_onnx::OBBBox obb(100, 100, 40, 20, 0.0f, 0.9f, 0);
    auto corners = obb.corners();
    TEST("OBB corners count", corners.size() == 4);
    // For angle=0, axis-aligned: cx=100,cy=100, w=40,h=20
    // corners: (80,90), (120,90), (120,110), (80,110)
    TEST("OBB corner[0] x", std::abs(corners[0].x - 80) < 1e-4);
    TEST("OBB corner[0] y", std::abs(corners[0].y - 90) < 1e-4);
    TEST("OBB corner[1] x", std::abs(corners[1].x - 120) < 1e-4);
    TEST("OBB corner[1] y", std::abs(corners[1].y - 90) < 1e-4);
    TEST("OBB corner[2] x", std::abs(corners[2].x - 120) < 1e-4);
    TEST("OBB corner[2] y", std::abs(corners[2].y - 110) < 1e-4);
    TEST("OBB corner[3] x", std::abs(corners[3].x - 80) < 1e-4);
    TEST("OBB corner[3] y", std::abs(corners[3].y - 110) < 1e-4);

    // OBBBox aabb
    auto aabb = obb.aabb();
    TEST("OBB aabb x1", std::abs(aabb.x1 - 80) < 1e-4);
    TEST("OBB aabb y1", std::abs(aabb.y1 - 90) < 1e-4);
    TEST("OBB aabb x2", std::abs(aabb.x2 - 120) < 1e-4);
    TEST("OBB aabb y2", std::abs(aabb.y2 - 110) < 1e-4);
}

void test_scale_funcs() {
    std::cout << "\n=== Test: Scale Functions ===\n" << std::endl;

    yolo_onnx::LetterboxInfo info = {0.5f, 10, 10, 640, 480, 640, 640};

    // Scale box
    yolo_onnx::Box box(10, 10, 100, 100, 1.0f, 0);
    yolo_onnx::scale_box(box, info);
    TEST("scale_box x1", std::abs(box.x1 - 0) < 1e-5);
    TEST("scale_box y1", std::abs(box.y1 - 0) < 1e-5);
    TEST("scale_box x2", std::abs(box.x2 - 180) < 1e-5);
    TEST("scale_box y2", std::abs(box.y2 - 180) < 1e-5);

    // Scale keypoint
    yolo_onnx::Keypoint kp{50, 50, 2.0f};
    yolo_onnx::scale_keypoint(kp, info);
    TEST("scale_keypoint x", std::abs(kp.x - 80) < 1e-5);
    TEST("scale_keypoint y", std::abs(kp.y - 80) < 1e-5);

    // Scale OBB box
    yolo_onnx::OBBBox obb(100, 100, 40, 20, 0.0f, 0.9f, 0);
    yolo_onnx::scale_obb_box(obb, info);
    TEST("scale_obb_box cx", std::abs(obb.cx - 180) < 1e-5);
    TEST("scale_obb_box cy", std::abs(obb.cy - 180) < 1e-5);
    TEST("scale_obb_box w", std::abs(obb.w - 80) < 1e-5);
    TEST("scale_obb_box h", std::abs(obb.h - 40) < 1e-5);
}

void test_task_type_names() {
    std::cout << "\n=== Test: Task Type Names ===\n" << std::endl;

    TEST("TaskType Detect", std::string(yolo_onnx::task_type_name(yolo_onnx::TaskType::Detect)) == "detect");
    TEST("TaskType Segment", std::string(yolo_onnx::task_type_name(yolo_onnx::TaskType::Segment)) == "segment");
    TEST("TaskType Pose", std::string(yolo_onnx::task_type_name(yolo_onnx::TaskType::Pose)) == "pose");
    TEST("TaskType OBB", std::string(yolo_onnx::task_type_name(yolo_onnx::TaskType::OBB)) == "obb");
}

int main() {
    std::cout << "=== yolo-onnx Unit Tests ===\n" << std::endl;

    test_types();
    test_model_creation();
    test_task_factory();
    test_infer_result_types();
    test_obb_result();
    test_scale_funcs();
    test_task_type_names();

    std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n" << std::endl;
    return (pass_count == test_count) ? 0 : 1;
}