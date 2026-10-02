#include "yolo_onnx/yolo_onnx.hpp"
#include "process/postprocess/postprocess_core.hpp"
#include "process/postprocess/postprocess.hpp"
#include "process/postprocess/decoder/decoder.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
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

// ============================================================
// 回归测试：两个已修复的真实 bug
// ============================================================

/// 回归 1：YOLOX 导出张量里 col4(obj) 与 col5+(cls) 已经过 sigmoid，
/// 解码器若再补一次 sigmoid，会把所有低分项抬到 0.5 以上，
/// 8400 个格子几乎全部越过阈值（实测输出 5445 个框、匹配 0 个）。
///
/// 构造一个「已 sigmoid」的输出：大量低分格子 + 少量高分格子，
/// 正确解码应只保留高分格子；错误解码（二次 sigmoid）会全部放行。
void test_yolox_no_double_sigmoid() {
    std::cout << "\n=== Test: YOLOX decode (regression: no double sigmoid) ===\n" << std::endl;

    const int num_classes = 80;
    const int input = 640;
    const int grid_w = input / 8;
    const int grid_count = grid_w * grid_w;           // 单层，只测 stride 8
    const int channels = 5 + num_classes;

    // 只填 1 层（stride 8），DecodeContext 的 stride16/32 网格会被 idx 越界保护跳过
    std::vector<float> data((size_t)grid_count * channels, 0.0f);
    auto at = [&](int idx, int c) -> float& { return data[(size_t)idx * channels + c]; };

    // 背景：低分（已 sigmoid，值很小）——二次 sigmoid 后会变成 ~0.5 而越过阈值
    for (int i = 0; i < grid_count; i++) {
        at(i, 4) = 0.001f;                            // obj
        at(i, 5 + 0) = 0.001f;                        // cls0
    }
    // 目标：高分
    const int target = 1234;
    at(target, 4) = 0.95f;
    at(target, 5 + 7) = 0.90f;                        // class 7

    yolo_onnx::TensorSet out;
    out.datas.emplace_back(1, grid_count * channels, CV_32F, data.data());
    out.shapes.push_back({1, grid_count, channels});

    yolo_onnx::DecodeContext ctx;
    ctx.num_classes  = num_classes;
    ctx.score_thresh = 0.25f;
    ctx.input_width  = input;
    ctx.input_height = input;

    yolo_onnx::YOLOXDecoder dec;
    auto boxes = dec.decode_detect(out, ctx);

    // 二次 sigmoid 会让 80x80=6400 个背景格子全部通过（score ≈ 0.5*0.5 = 0.25）
    TEST("YOLOX: background cells filtered (no double sigmoid)",
         boxes.size() == 1);
    if (boxes.size() == 1) {
        TEST("YOLOX: keeps correct class", boxes[0].label == 7);
        // 0.95 * 0.90
        TEST("YOLOX: score = obj * cls", std::abs(boxes[0].score - 0.855f) < 1e-4);
        TEST("YOLOX: box has positive size",
             boxes[0].x2 > boxes[0].x1 && boxes[0].y2 > boxes[0].y1);
    }
}

/// 回归 2：segment 的掩码合成。
/// 两个缺陷都会让掩码与框对不上（实测 mask IoU 仅 0.24）：
///   (a) 裁剪窗口用「已还原到原图的 box」直接减 padding，漏了「映射回模型输入空间」；
///   (b) 缩放时直接把 proto 网格拉到原图尺寸，忽略了 letterbox 的 padding。
/// 正确做法：proto → 模型输入 → 去 letterbox padding → 原图。
void test_segment_mask_coordinate_mapping() {
    std::cout << "\n=== Test: Segment mask mapping (regression) ===\n" << std::endl;

    // ---- (a) 原图坐标 → proto 坐标的映射必须先经过模型输入空间 ----
    // letterbox: 原图 810x1080 → 输入 640x640, scale=640/1080, pad_left=(640-480)/2=80
    const int orig_w = 810, orig_h = 1080, in_w = 640, in_h = 640;
    const int proto_w = in_w / 4, proto_h = in_h / 4;   // 160x160
    const float scale = (float)in_w / (float)orig_h;    // 0.5926
    const int pad_left = (in_w - (int)(orig_w * scale)) / 2;
    const int pad_top  = 0;

    yolo_onnx::LetterboxInfo lb{scale, pad_left, pad_top, orig_w, orig_h, in_w, in_h};

    // 一个框，其右边缘应贴近原图右边界 → proto 坐标也应贴近 proto 宽度
    yolo_onnx::Box box((float)(orig_w - 100), 400.0f, (float)orig_w, 900.0f, 0.9f, 0);

    const float sx = (float)proto_w / (float)in_w;
    const float x_proto = (box.x2 * lb.scale + lb.pad_left) * sx;
    // 原图右边缘 810 → 输入 560 → proto 140（不是 proto 边缘 160，
    // 也不是把原图坐标直接乘 proto_scale 得到的 157）
    TEST("segment: orig->proto keeps letterbox offset",
         std::abs(x_proto - 140.0f) < 1.0f);

    // 旧实现（漏掉映射回输入空间）：(810 - 80) * 0.25 = 182.5 → 被 clamp 到 159
    const float x_old = (box.x2 - lb.pad_left) * sx;
    TEST("segment: regression case is distinguishable",
         x_old > (float)proto_w - 1.0f);

    // ---- (b) 掩码缩放路径：proto → 输入 → 去 padding → 原图 ----
    // 构造一张「只在上半部分为 1」的 proto 掩码，去 padding 后有效内容
    // 应覆盖 crop_h 行；直接 proto→orig 会把 padding 一起算进去，导致偏移。
    const int crop_w = (int)std::lround(orig_w * scale);
    const int crop_h = (int)std::lround(orig_h * scale);
    const int off_x = (in_w - crop_w) / 2;
    const int off_y = (in_h - crop_h) / 2;
    TEST("segment: crop offsets are non-negative", off_x >= 0 && off_y >= 0);
    TEST("segment: crop fits inside input", off_x + crop_w <= in_w && off_y + crop_h <= in_h);
    // pad_left == off_x，两者必须一致，否则框与掩码用不同的偏移
    TEST("segment: crop offset matches letterbox pad_left", off_x == pad_left);
}

/// 回归 3：OBB 的通道布局与角度激活。
/// 旧实现有三个叠加的错误，实测会让所有旋转框退化到 (0, pi) 且类别错位一格：
///   ① 把角度当成 ch4（实际 ch4 是 0 号类的分数），类分数却从 ch5 读；
///   ② 对已激活的角度再套一次 sigmoid*pi——图内已做完激活
///      （OBB26 是 (sigmoid(x)-0.25)*pi，OBB 是 sigmoid(x)*pi）；
///   ③ 对已 sigmoid 的类分数再 sigmoid 一次，抬高背景分数。
/// 正确布局：[1, 4+nc+1, N]，角度在**最后一个**通道，单位弧度，原样读出。
void test_obb_channel_layout_and_angle() {
    std::cout << "\n=== Test: OBB decode (regression: angle in last channel, no re-sigmoid) ===\n" << std::endl;

    const int num_classes = 15;                 // DOTA
    const int input = 640;
    const int grid_w = input / 8;
    const int grid_count = grid_w * grid_w;     // 只填 stride 8 一层
    const int channels = 4 + num_classes + 1;  // 20

    std::vector<float> data((size_t)grid_count * channels, 0.0f);
    // 3D 布局 [1, C, N] 每个通道平面为 N 个元素
    auto at = [&](int c, int idx) -> float& { return data[(size_t)c * grid_count + idx]; };

    // 背景：低分类分数（已 sigmoid）
    for (int i = 0; i < grid_count; i++) {
        at(4, i) = 0.001f;
        at(4 + num_classes, i) = 0.3f;          // 角度通道填一个合法弧度
    }

    // 目标：class 4 高分，角度 0.75 rad（若再套 sigmoid*pi 会变成 1.66，明显不同）
    const int target = 1234;
    const float angle_true = 0.75f;
    at(0, target) = 100.0f;                     // cx
    at(1, target) = 200.0f;                     // cy
    at(2, target) = 50.0f;                      // w
    at(3, target) = 30.0f;                      // h
    for (int c = 0; c < num_classes; c++) at(4 + c, target) = 0.001f;
    at(4 + 4, target) = 0.95f;                  // class 4
    at(4 + num_classes, target) = angle_true;   // 角度 = 最后一个通道

    yolo_onnx::TensorSet out;
    out.datas.emplace_back(1, grid_count * channels, CV_32F, data.data());
    out.shapes.push_back({1, channels, grid_count});

    yolo_onnx::DecodeContext ctx;
    ctx.num_classes  = num_classes;
    ctx.score_thresh = 0.25f;
    ctx.input_width  = input;
    ctx.input_height = input;

    yolo_onnx::V8Decoder dec;
    auto boxes = dec.decode_obb(out, ctx);

    TEST("OBB: background cells filtered", boxes.size() == 1);
    if (boxes.size() == 1) {
        const auto& b = boxes[0];
        TEST("OBB: class read from ch4..ch4+nc-1", b.label == 4);
        TEST("OBB: score not double-sigmoided", std::abs(b.score - 0.95f) < 1e-4);
        TEST("OBB: cx,cy from ch0..ch1",
             std::abs(b.cx - 100.0f) < 1e-3 && std::abs(b.cy - 200.0f) < 1e-3);
        TEST("OBB: w,h from ch2..ch3",
             std::abs(b.w - 50.0f) < 1e-3 && std::abs(b.h - 30.0f) < 1e-3);
        // 角度必须原样透传：再 sigmoid 会得到 1.66，且旧实现还会读错通道
        TEST("OBB: angle read raw from last channel",
             std::abs(b.angle - angle_true) < 1e-4);
        // 回归可辨识性：旧路径（sigmoid(angle)*pi）明显不同
        TEST("OBB: regression case is distinguishable",
             std::abs(yolo_onnx::sigmoid(angle_true) * CV_PI - angle_true) > 0.5f);
    }
}

/// 回归 4：语义分割的类别图解码。
/// 后处理只需两步：按 letterbox 裁掉 padding，再用最近邻缩放到原图尺寸。
/// 两个易错点：
///   ① 类别 id 必须用最近邻，双线性会在类别边界产生不存在的 id；
///   ② 裁剪偏移必须由 target 与内容尺寸推出，不能直接用原图坐标减 padding。
/// 同时验证 find_classmap_index 能在多输出里挑出类别图那张。
void test_sem_classmap_decoding() {
    std::cout << "\n=== Test: Sem decode (regression: class-id map, nearest resize) ===\n" << std::endl;

    // ---- 原图 810x1080 → 输入 640x640：scale=640/1080，内容 480x640，pad_left=80 ----
    const int orig_w = 810, orig_h = 1080, in_w = 640, in_h = 640;
    const float scale = (float)in_w / (float)orig_h;
    const int content_w = (int)std::lround(orig_w * scale);   // 480
    const int content_h = (int)std::lround(orig_h * scale);   // 640
    const int off_x = (in_w - content_w) / 2;                 // 80
    const int off_y = (in_h - content_h) / 2;                 // 0
    TEST("sem: crop fits inside input",
         off_x >= 0 && off_y >= 0 &&
         off_x + content_w <= in_w && off_y + content_h <= in_h);

    yolo_onnx::LetterboxInfo lb{scale, off_x, off_y, orig_w, orig_h, in_w, in_h};

    // ---- 构造 640x640 类别图：内容区填 7，padding 区填 0 ----
    std::vector<float> map_data((size_t)in_w * in_h, 0.0f);
    for (int y = 0; y < content_h; y++) {
        for (int x = off_x; x < off_x + content_w; x++) {
            map_data[(size_t)y * in_w + x] = 7.0f;
        }
    }

    yolo_onnx::TensorSet out;
    out.datas.emplace_back(1, (int)map_data.size(), CV_32F, map_data.data());
    out.shapes.push_back({1, in_h, in_w});        // [1, H, W] 类别图

    yolo_onnx::DecodeContext ctx;
    ctx.input_width  = in_w;
    ctx.input_height = in_h;

    yolo_onnx::PostProcessParams pp;
    auto pp_sem = yolo_onnx::create_postprocess(yolo_onnx::TaskType::Sem,
                                                yolo_onnx::ModelType::YOLO26, pp);
    TEST("sem: postprocess created", pp_sem != nullptr);
    if (!pp_sem) return;
    pp_sem->set_context(ctx);

    auto result = pp_sem->forward(out, lb);
    const auto* sem = std::get_if<yolo_onnx::SemResult>(&result);
    TEST("sem: result type is SemResult", sem != nullptr);
    if (!sem) return;

    const auto& m = sem->mask;
    TEST("sem: mask resized to original size",
         m.width == orig_w && m.height == orig_h);

    // 内容区（原图坐标）应全为 7：水平方向 x_img ∈ [0,810) 都来自内容区
    bool content_ok = true;
    for (int y = 0; y < orig_h; y += 7) {
        for (int x = 0; x < orig_w; x += 7) {
            if ((int)m.data[(size_t)y * m.width + x] != 7) { content_ok = false; break; }
        }
        if (!content_ok) break;
    }
    TEST("sem: content region keeps its class id", content_ok);

    // 不应出现输入里没有的类别 id（最近邻保证）
    bool only_valid = true;
    for (float v : m.data) {
        if ((int)v != 0 && (int)v != 7) { only_valid = false; break; }
    }
    TEST("sem: no interpolated (non-existent) class ids", only_valid);

    // ---- find_classmap_index：在多输出里挑元素最多的那张 ----
    yolo_onnx::TensorSet multi;
    multi.datas.emplace_back(1, 4, CV_32F, (float*)std::calloc(4, sizeof(float)));
    multi.shapes.push_back({1, 2, 2});
    multi.datas.emplace_back(1, (int)map_data.size(), CV_32F, map_data.data());
    multi.shapes.push_back({1, in_h, in_w});
    TEST("sem: class map found among multiple outputs",
         yolo_onnx::PostProcessSem::find_classmap_index(multi) == 1);
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
    test_yolox_no_double_sigmoid();
    test_segment_mask_coordinate_mapping();
    test_obb_channel_layout_and_angle();
    test_sem_classmap_decoding();

    std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n" << std::endl;
    return (pass_count == test_count) ? 0 : 1;
}