#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "yolo_onnx/yolo_onnx.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <cstdint>

namespace py = pybind11;
namespace yo = yolo_onnx;

// ============================================================
// Helper: convert numpy array (H, W, C) uint8 BGR to cv::Mat
// ============================================================
static cv::Mat numpy_to_cvmat(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Expected 3-dimensional array (H, W, C)");
    }
    int h = (int)buf.shape[0];
    int w = (int)buf.shape[1];
    int c = (int)buf.shape[2];
    if (c != 3) {
        throw std::runtime_error("Expected 3 channels (BGR), got " + std::to_string(c) + " channels");
    }
    return cv::Mat(h, w, CV_8UC3, buf.ptr).clone();
}

// ============================================================
// pybind11 module: yolo_onnx
// ============================================================
PYBIND11_MODULE(yolo_onnx_py, m) {
    m.doc() = "YOLO inference framework based on ONNX Runtime — Python bindings";

    // ---- Enums ----
    py::enum_<yo::ModelType>(m, "ModelType")
        .value("YOLOv5",   yo::ModelType::YOLOv5)
        .value("YOLOX",    yo::ModelType::YOLOX)
        .value("YOLOv8",   yo::ModelType::YOLOv8)
        .value("YOLOv11",  yo::ModelType::YOLOv11)
        .value("YOLO26",   yo::ModelType::YOLO26)
        .value("PPYOLOE",  yo::ModelType::PPYOLOE)
        .export_values();

    py::enum_<yo::TaskType>(m, "TaskType")
        .value("Detect",   yo::TaskType::Detect)
        .value("Segment",  yo::TaskType::Segment)
        .value("Pose",     yo::TaskType::Pose)
        .value("OBB",      yo::TaskType::OBB)
        .export_values();

    // ---- Data types ----
    py::class_<yo::Box>(m, "Box")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, int>(),
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("score"), py::arg("label"))
        .def_readwrite("x1",    &yo::Box::x1)
        .def_readwrite("y1",    &yo::Box::y1)
        .def_readwrite("x2",    &yo::Box::x2)
        .def_readwrite("y2",    &yo::Box::y2)
        .def_readwrite("score", &yo::Box::score)
        .def_readwrite("label", &yo::Box::label)
        .def("width",  &yo::Box::width)
        .def("height", &yo::Box::height)
        .def("area",   &yo::Box::area)
        .def("__repr__", [](const yo::Box& b) {
            return "<Box label=" + std::to_string(b.label) +
                   " score=" + std::to_string(b.score) +
                   " [" + std::to_string((int)b.x1) + "," + std::to_string((int)b.y1) +
                   "," + std::to_string((int)b.x2) + "," + std::to_string((int)b.y2) + "]>";
        });

    py::class_<yo::Keypoint>(m, "Keypoint")
        .def(py::init<>())
        .def_readwrite("x",          &yo::Keypoint::x)
        .def_readwrite("y",          &yo::Keypoint::y)
        .def_readwrite("visibility", &yo::Keypoint::visibility)
        .def("__repr__", [](const yo::Keypoint& kp) {
            return "<Keypoint (" + std::to_string(kp.x) + ", " + std::to_string(kp.y) +
                   ") vis=" + std::to_string(kp.visibility) + ">";
        });

    py::class_<yo::Mask>(m, "Mask")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def_readwrite("width",  &yo::Mask::width)
        .def_readwrite("height", &yo::Mask::height)
        .def_readwrite("data",   &yo::Mask::data)
        .def("__repr__", [](const yo::Mask& mask) {
            return "<Mask " + std::to_string(mask.width) + "x" + std::to_string(mask.height) + ">";
        })
        .def("to_numpy", [](const yo::Mask& mask) {
            // Return a 2D numpy array (H, W) of float32
            py::array_t<float> arr({mask.height, mask.width});
            auto buf = arr.request();
            std::memcpy(buf.ptr, mask.data.data(), mask.data.size() * sizeof(float));
            return arr;
        });

    py::class_<yo::OBBBox>(m, "OBBBox")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, float, int>(),
             py::arg("cx"), py::arg("cy"), py::arg("w"), py::arg("h"),
             py::arg("angle"), py::arg("score"), py::arg("label"))
        .def_readwrite("cx",    &yo::OBBBox::cx)
        .def_readwrite("cy",    &yo::OBBBox::cy)
        .def_readwrite("w",     &yo::OBBBox::w)
        .def_readwrite("h",     &yo::OBBBox::h)
        .def_readwrite("angle", &yo::OBBBox::angle)
        .def_readwrite("score", &yo::OBBBox::score)
        .def_readwrite("label", &yo::OBBBox::label)
        .def("corners", &yo::OBBBox::corners)
        .def("aabb",    &yo::OBBBox::aabb)
        .def("__repr__", [](const yo::OBBBox& obb) {
            return "<OBBBox label=" + std::to_string(obb.label) +
                   " score=" + std::to_string(obb.score) +
                   " center=(" + std::to_string((int)obb.cx) + "," + std::to_string((int)obb.cy) +
                   ") size=" + std::to_string((int)obb.w) + "x" + std::to_string((int)obb.h) + ">";
        });

    py::class_<yo::LetterboxInfo>(m, "LetterboxInfo")
        .def(py::init<>())
        .def_readwrite("scale",    &yo::LetterboxInfo::scale)
        .def_readwrite("pad_left", &yo::LetterboxInfo::pad_left)
        .def_readwrite("pad_top",  &yo::LetterboxInfo::pad_top)
        .def_readwrite("orig_w",   &yo::LetterboxInfo::orig_w)
        .def_readwrite("orig_h",   &yo::LetterboxInfo::orig_h)
        .def_readwrite("target_w", &yo::LetterboxInfo::target_w)
        .def_readwrite("target_h", &yo::LetterboxInfo::target_h);

    // ---- Result types ----
    py::class_<yo::DetectResult>(m, "DetectResult")
        .def(py::init<>())
        .def_readwrite("boxes", &yo::DetectResult::boxes)
        .def("__repr__", [](const yo::DetectResult& r) {
            return "<DetectResult " + std::to_string(r.boxes.size()) + " boxes>";
        });

    py::class_<yo::SegmentResult>(m, "SegmentResult")
        .def(py::init<>())
        .def_readwrite("boxes", &yo::SegmentResult::boxes)
        .def_readwrite("masks", &yo::SegmentResult::masks)
        .def("__repr__", [](const yo::SegmentResult& r) {
            return "<SegmentResult " + std::to_string(r.boxes.size()) + " boxes, " +
                   std::to_string(r.masks.size()) + " masks>";
        });

    py::class_<yo::PoseResult>(m, "PoseResult")
        .def(py::init<>())
        .def_readwrite("boxes",    &yo::PoseResult::boxes)
        .def_readwrite("keypoints", &yo::PoseResult::keypoints)
        .def("__repr__", [](const yo::PoseResult& r) {
            return "<PoseResult " + std::to_string(r.boxes.size()) + " poses>";
        });

    py::class_<yo::OBBResult>(m, "OBBResult")
        .def(py::init<>())
        .def_readwrite("obb_boxes", &yo::OBBResult::obb_boxes)
        .def("__repr__", [](const yo::OBBResult& r) {
            return "<OBBResult " + std::to_string(r.obb_boxes.size()) + " obb_boxes>";
        });

    // ---- Model::Config ----
    py::class_<yo::Model::Config>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_path",    &yo::Model::Config::model_path)
        .def_readwrite("model_type",    &yo::Model::Config::model_type)
        .def_readwrite("task_type",     &yo::Model::Config::task_type)
        .def_readwrite("backend_type",  &yo::Model::Config::backend_type)
        .def_readwrite("score_thresh",  &yo::Model::Config::score_thresh)
        .def_readwrite("nms_thresh",    &yo::Model::Config::nms_thresh)
        .def_readwrite("input_width",   &yo::Model::Config::input_width)
        .def_readwrite("input_height",  &yo::Model::Config::input_height)
        .def_readwrite("num_classes",   &yo::Model::Config::num_classes)
        .def_readwrite("num_keypoints", &yo::Model::Config::num_keypoints)
        .def_readwrite("num_threads",   &yo::Model::Config::num_threads)
        .def("__repr__", [](const yo::Model::Config& c) {
            return "<ModelConfig " + c.model_path +
                   " type=" + yo::model_type_name(c.model_type) +
                   " task=" + yo::task_type_name(c.task_type) + ">";
        });

    // ---- Model base class ----
    // 模型不再按版本派生子类：detect/segment/pose/obb 全部由同一个 Model 承担，
    // 任务差异由 config_.task_type 决定，入口按返回类型区分。
    py::class_<yo::Model, std::shared_ptr<yo::Model>>(m, "Model")
        .def(py::init<>())
        .def("load", &yo::Model::load, py::arg("config"))
        .def("config", &yo::Model::config, py::return_value_policy::reference_internal)
        .def("backend", &yo::Model::backend)
        .def("infer", [](yo::Model& self, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
            cv::Mat mat = numpy_to_cvmat(img);
            return self.infer(mat);
        }, py::arg("image"), "Run inference on a numpy image (H, W, 3) uint8 BGR; "
                             "result type follows config.task_type")
        .def("infer_file", [](yo::Model& self, const std::string& path) {
            cv::Mat mat = cv::imread(path);
            if (mat.empty()) {
                throw std::runtime_error("Failed to load image: " + path);
            }
            return self.infer(mat);
        }, py::arg("path"), "Run inference on an image file")
        .def("infer_detect", [](yo::Model& self,
                                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
            return self.infer_detect(numpy_to_cvmat(img));
        }, py::arg("image"), "Run detection inference, returns DetectResult")
        .def("infer_segment", [](yo::Model& self,
                                 py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
            return self.infer_segment(numpy_to_cvmat(img));
        }, py::arg("image"), "Run segmentation inference, returns SegmentResult")
        .def("infer_segment_file", [](yo::Model& self, const std::string& path) {
            cv::Mat mat = cv::imread(path);
            if (mat.empty()) {
                throw std::runtime_error("Failed to load image: " + path);
            }
            return self.infer_segment(mat);
        }, py::arg("path"), "Run segmentation inference on an image file")
        .def("infer_pose", [](yo::Model& self,
                             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
            return self.infer_pose(numpy_to_cvmat(img));
        }, py::arg("image"), "Run pose inference, returns PoseResult")
        .def("infer_pose_file", [](yo::Model& self, const std::string& path) {
            cv::Mat mat = cv::imread(path);
            if (mat.empty()) {
                throw std::runtime_error("Failed to load image: " + path);
            }
            return self.infer_pose(mat);
        }, py::arg("path"), "Run pose inference on an image file")
        .def("infer_obb", [](yo::Model& self,
                            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
            return self.infer_obb(numpy_to_cvmat(img));
        }, py::arg("image"), "Run OBB inference, returns OBBResult")
        .def("infer_obb_file", [](yo::Model& self, const std::string& path) {
            cv::Mat mat = cv::imread(path);
            if (mat.empty()) {
                throw std::runtime_error("Failed to load image: " + path);
            }
            return self.infer_obb(mat);
        }, py::arg("path"), "Run OBB inference on an image file");

    // ---- Factory functions ----
    m.def("create_model", py::overload_cast<yo::ModelType>(&yo::create_model),
          py::arg("type"), "Create a detection model by type");
    m.def("create_model", py::overload_cast<yo::ModelType, yo::TaskType>(&yo::create_model),
          py::arg("type"), py::arg("task"),
          "Create a model by type and task (detect/segment/pose/obb)");

    // ---- Utility functions ----
    m.def("model_type_name", &yo::model_type_name, py::arg("type"),
          "Get human-readable name for a ModelType");
    m.def("task_type_name", &yo::task_type_name, py::arg("type"),
          "Get human-readable name for a TaskType");
    m.def("iou", &yo::iou, py::arg("a"), py::arg("b"),
          "Compute IoU between two boxes");
    m.def("nms", &yo::nms, py::arg("boxes"), py::arg("iou_threshold"),
          "CPU Non-Maximum Suppression, returns indices of kept boxes");
    m.def("sigmoid", &yo::sigmoid, py::arg("x"),
          "Sigmoid function");

    // ---- Top-level convenience: run inference in one step ----
    m.def("detect", [](const std::string& model_path,
                       py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
                       const std::string& model_type_str,
                       float score_thresh,
                       float nms_thresh,
                       int input_size,
                       int num_classes,
                       int num_threads) {
        // Map model type string
        yo::ModelType model_type;
        if (model_type_str == "v5")         model_type = yo::ModelType::YOLOv5;
        else if (model_type_str == "yolox") model_type = yo::ModelType::YOLOX;
        else if (model_type_str == "v8")    model_type = yo::ModelType::YOLOv8;
        else if (model_type_str == "v11")   model_type = yo::ModelType::YOLOv11;
        else if (model_type_str == "v26")   model_type = yo::ModelType::YOLO26;
        else if (model_type_str == "ppyoloe") model_type = yo::ModelType::PPYOLOE;
        else throw std::runtime_error("Unknown model type: " + model_type_str);

        auto model = yo::create_model(model_type);
        yo::Model::Config config;
        config.model_path    = model_path;
        config.model_type    = model_type;
        config.score_thresh  = score_thresh;
        config.nms_thresh    = nms_thresh;
        config.input_width   = input_size;
        config.input_height  = input_size;
        config.num_classes   = num_classes;
        config.num_threads   = num_threads;

        if (!model->load(config)) {
            throw std::runtime_error("Failed to load model: " + model_path);
        }

        cv::Mat mat = numpy_to_cvmat(img);
        return model->infer(mat);
    }, py::arg("model_path"), py::arg("image"),
       py::arg("model_type_str") = "v8",
       py::arg("score_thresh") = 0.5f,
       py::arg("nms_thresh") = 0.45f,
       py::arg("input_size") = 640,
       py::arg("num_classes") = 80,
       py::arg("num_threads") = 4,
       "One-shot detection: load model, infer, return boxes");

    m.def("detect_file", [](const std::string& model_path,
                            const std::string& image_path,
                            const std::string& model_type_str,
                            float score_thresh,
                            float nms_thresh,
                            int input_size,
                            int num_classes,
                            int num_threads) {
        cv::Mat mat = cv::imread(image_path);
        if (mat.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }
        // Map model type string
        yo::ModelType model_type;
        if (model_type_str == "v5")         model_type = yo::ModelType::YOLOv5;
        else if (model_type_str == "yolox") model_type = yo::ModelType::YOLOX;
        else if (model_type_str == "v8")    model_type = yo::ModelType::YOLOv8;
        else if (model_type_str == "v11")   model_type = yo::ModelType::YOLOv11;
        else if (model_type_str == "v26")   model_type = yo::ModelType::YOLO26;
        else if (model_type_str == "ppyoloe") model_type = yo::ModelType::PPYOLOE;
        else throw std::runtime_error("Unknown model type: " + model_type_str);

        auto model = yo::create_model(model_type);
        yo::Model::Config config;
        config.model_path    = model_path;
        config.model_type    = model_type;
        config.score_thresh  = score_thresh;
        config.nms_thresh    = nms_thresh;
        config.input_width   = input_size;
        config.input_height  = input_size;
        config.num_classes   = num_classes;
        config.num_threads   = num_threads;

        if (!model->load(config)) {
            throw std::runtime_error("Failed to load model: " + model_path);
        }

        return model->infer(mat);
    }, py::arg("model_path"), py::arg("image_path"),
       py::arg("model_type_str") = "v8",
       py::arg("score_thresh") = 0.5f,
       py::arg("nms_thresh") = 0.45f,
       py::arg("input_size") = 640,
       py::arg("num_classes") = 80,
       py::arg("num_threads") = 4,
       "One-shot detection from file: load model, infer, return boxes");
}