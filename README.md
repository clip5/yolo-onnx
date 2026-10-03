# yolo-onnx

English | [简体中文](README_ZH.md)

A C++17 YOLO inference framework built on ONNX Runtime. One `.onnx` file all the way through; hardware acceleration is switched via Execution Providers, with zero EP dependencies at compile time.

[![C++17](https://img.shields.io/badge/C++-17-00599C.svg)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.12%2B-00599C.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-00599C.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

<div align="center">
  <img src="assets/results/detect_bus.jpg" width="24%">
  <img src="assets/results/segment_bus.jpg" width="24%">
  <img src="assets/results/pose_bus.jpg" width="24%">
  <img src="assets/results/sem_bus.jpg" width="24%">
  <p><em>YOLO26 detection · segmentation · pose estimation · semantic segmentation — all actual inference outputs of this framework on bus.jpg</em></p>
</div>

## ✨ Features

- 🧩 **Wide model coverage**: YOLOv5 / YOLOX / YOLOv8 / YOLOv11 / YOLO26 / PPYOLOE — one `Decoder` subclass per export format, fully independent of each other
- 🚀 **Unified multi-task interface**: detection / segmentation / pose / rotated boxes / semantic segmentation share one pipeline, returned as `std::variant` with no `dynamic_cast`
- ⚡ **Plug-and-play EPs**: CUDA / TensorRT / OpenVINO / ROCm / MIGraphX / VitisAI / QNN / CoreML / XNNPACK… switching EPs is a one-parameter change, **no recompilation needed** (EP shared libraries are loaded at runtime by onnxruntime)
- 🧪 **Verifiable accuracy**: `tools/` provides a toolchain for per-box comparison against ultralytics / onnxruntime reference implementations — accuracy issues are settled with numbers, not eyeballing images
- 🪶 **Zero hidden dependencies**: only two public headers (`yolo_onnx.hpp` + `yolo_onnx_types.hpp`); backend, pre/post-processing all live inside `src/`
- 🔌 **Independently reusable**: the NMS / IoU / coordinate-restore free functions in `postprocess_core.hpp` are header-only and can be copied straight into other projects

## 📊 Benchmarks

Measured on **RTX 4070 Ti SUPER** + **i5-10400F (12 threads)** with `yolo11n` at 640×640, batch=1, averaged over 100 runs. The numbers are **end-to-end `Model::infer()`** (letterbox preprocessing, NMS, and everything before visualization included):

| Backend | End-to-end latency | Throughput | vs CPU |
|------|-----------|------|---------|
| ONNX Runtime CPU | 47.6 ms | 21 fps | 1.0× |
| OpenVINO CPU | 32.1 ms | 31 fps | 1.5× |
| OpenVINO GPU (Intel iGPU) | 14.3 ms | 70 fps | 3.3× |
| **ONNX Runtime CUDA EP** | **8.1 ms** | **123 fps** | **5.8×** |
| **ONNX Runtime TensorRT EP (FP16)** | **7.1 ms** | **141 fps** | **6.7×** |

> The TensorRT EP builds an engine on first load (~1 min); engine caching (`/tmp`) is enabled by default, so subsequent loads are instant.
> Figures above were measured end-to-end via `Model::infer()` (preprocess + inference + decode + NMS), fixed 1920×1080 input, batch=1, averaged over 100 runs.

## 📑 Table of Contents

- [yolo-onnx](#yolo-onnx)
  - [✨ Features](#-features)
  - [📊 Benchmarks](#-benchmarks)
  - [🚀 Quick Start](#-quick-start)
  - [🧩 C++ Usage](#-c-usage)
  - [🧠 Architecture](#-architecture)
  - [🔌 Backends & Execution Providers](#-backends--execution-providers)
  - [🧪 Accuracy Verification Tools](#-accuracy-verification-tools)
  - [🧱 Extending](#-extending)
  - [📁 Project Layout](#-project-layout)
  - [📚 Documentation](#-documentation)

## 🚀 Quick Start

### Dependencies

- C++17 compiler
- OpenCV >= 4.x
- ONNX Runtime >= 1.12 (CPU build is enough for all features; GPU acceleration needs the official `onnxruntime-gpu` package)

### Build

```bash
git clone <this-repo> && cd yolo-onnx

# Point to your ONNX Runtime location (or use the ONNXRUNTIME_DIR env var)
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.23.2

mkdir build && cd build
cmake .. -DWITH_EXAMPLES=ON -DWITH_TESTS=ON
make -j$(nproc)

./tests/test_yolo        # 84 unit tests
```

CMake options (all OFF by default unless noted):

| Option | Description |
|------|------|
| `WITH_EXAMPLES` | Build `examples/` (**ON** by default) |
| `WITH_TESTS` | Build `tests/test_yolo` |
| `WITH_TOOLS` | Build `tools/dump_json` (for accuracy comparison) |
| `WITH_PYTHON` | Build pybind11 bindings (`python/`) |

### Command-line Inference

```bash
# Detection (CPU by default)
./examples/detect_image v11 ../assets/models/yolo11n.onnx ../assets/images/bus.jpg out.jpg

# Switch Execution Provider — same binary, no recompilation
./examples/detect_image v11 model.onnx bus.jpg out.jpg --ep=cuda
./examples/detect_image v11 model.onnx bus.jpg out.jpg --ep=tensorrt --fp16

# Segmentation
./examples/segment_image v11 ../assets/models/yolo11s-seg640.onnx bus.jpg out.jpg --ep=cuda

# Pose estimation
./examples/pose_image v26 ../assets/models/yolo26s-pose.onnx ../assets/images/bus.jpg out.jpg --ep=cuda

# Semantic segmentation (outputs a class-id map, colored with the ultralytics palette, optional legend)
./examples/sem_image v26 ../assets/models/yolo26s-sem.onnx ../assets/images/bus.jpg out.jpg --classes=19 --legend

# Adjust thresholds / input size / class count
./examples/detect_image v5 model.onnx in.jpg out.jpg --score=0.25 --nms=0.45 --size=640 --classes=80
```

Valid `model_type` values: `v5` / `yolox` / `v8` / `v11` / `v26` / `ppyoloe`.

Output:

```
Detected 5 objects:
  label=5 score=0.876215 rect=[4,260,805,751]
  label=0 score=0.875915 rect=[52,436,243,935]
  ...
```

## 🧩 C++ Usage

### Detection

```cpp
#include "yolo_onnx/yolo_onnx.hpp"
#include <opencv2/opencv.hpp>

auto model = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv11);

yolo_onnx::Model::Config config;
config.model_path    = "yolo11n.onnx";
config.score_thresh  = 0.5f;
config.nms_thresh    = 0.45f;
config.custom_config = "ep=cuda";        // switching EPs is a one-line change
model->load(config);

cv::Mat image = cv::imread("bus.jpg");
yolo_onnx::DetectResult det = model->infer_detect(image);

for (const auto& box : det.boxes) {
    printf("label=%d score=%.3f [%.0f,%.0f,%.0f,%.0f]\n",
           box.label, box.score, box.x1, box.y1, box.x2, box.y2);
}
```

### Multi-task: Segmentation / Pose / Rotated Boxes / Semantic Segmentation

All five tasks share one `Model`; results are returned as a `std::variant` and extracted safely with `std::get_if`:

```cpp
auto model = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv11,
                                     yolo_onnx::TaskType::Segment);
model->load(config);

yolo_onnx::InferResult result = model->infer(image);

if (auto* seg = std::get_if<yolo_onnx::SegmentResult>(&result)) {
    // seg->boxes[i] — detection boxes
    // seg->masks[i] — segmentation masks (already restored to original image size)
} else if (auto* pose = std::get_if<yolo_onnx::PoseResult>(&result)) {
    // pose->boxes[i].keypoints
} else if (auto* obb = std::get_if<yolo_onnx::OBBResult>(&result)) {
    // obb->obb_boxes[i] — rotated boxes (cx, cy, w, h, angle)
} else if (auto* sem = std::get_if<yolo_onnx::SemResult>(&result)) {
    // sem->mask — class-id map at original image size (one class id per pixel, not probabilities)
}
```

Typed entry points are also available: `infer_detect()` / `infer_segment()` / `infer_pose()` / `infer_obb()` / `infer_sem()`.

## 🧠 Architecture

The pipeline is a straight line; every stage sits behind an abstract interface, so **adding a new export format or task never touches the main path**:

```
image → PreProcess → Backend(forward) → TensorSet → Decoder → PostProcess → InferResult
        preprocessing  inference engine  unified tensors  decode  task postprocess  results
```

```
yolo_onnx
├── Backend (src/core/)              ← inference engine, EP switching lives here
│   └── OnnxruntimeBackend           ← main line: CPU / CUDA / TensorRT / OpenVINO / ...
│       └── (optional) OpenvinoBackend  ← standalone runtime, not compiled by default
├── PreProcess (src/process/preprocess/)
│   └── PreProcessParams::for_model()  ← all per-model preprocessing differences live here
├── Decoder (src/process/postprocess/decoder/)
│   └── V5Decoder / YOLOXDecoder / V8Decoder / PPYOLOEDecoder
├── PostProcess (src/process/postprocess/)
│   ├── postprocess_core.hpp          ← zero-dependency free functions (nms / iou / restore_* / detect_pipeline)
│   └── Detect / Segment / Pose / OBB ← task dispatch only, logic reused from core
└── Model (src/model.cpp)            ← single Model class + PIMPL, no subclasses
```

Three orthogonal design axes:

| What you want to add | Where to change | Anything else? |
|---------|-------|------------|
| New export format | Add a `Decoder` subclass + one factory registration line | No |
| New task | Add a `PostProcess` subclass + registration + result type | No |
| New EP | Add one case in `setup_execution_providers()` | No |

## 🔌 Backends & Execution Providers

**ONNX Runtime is the main line**: all hardware acceleration goes through EPs, and no EP library is linked at compile time. Switching EPs is a one-parameter change (`--ep=cuda`, `--ep=tensorrt --fp16`, `--ep=openvino` …) with the same binary — **no recompilation**. If an EP is unavailable, the reason is printed and it **falls back to CPU automatically**; inference never fails outright.

> Full EP list (dedicated-API vs generic whitelist), configuration options, standalone-runtime backends (OpenVINO / CANN / RKNN), and pitfalls encountered — see **[docs/backends.md](docs/backends.md)**

## 🧪 Accuracy Verification Tools

`tools/` provides a numeric accuracy-validation chain: **dump JSON → run reference implementation → per-box IoU comparison**. Accuracy issues are pinpointed with numbers instead of staring at annotated images (several historical decode bugs were completely invisible in images yet glaring in the IoU numbers).

> Three-step quickstart, usage of each script, mask validation (`check_masks_ul.py`), preprocessing sweeps, and historical bug post-mortems — see **[docs/accuracy.md](docs/accuracy.md)**

## 🧱 Extending

Three **orthogonal** extension axes, each independent:

| What you want to add | What to change | What stays untouched |
|---------|-----------|---------|
| New **export format** | Add a `Decoder` subclass + factory registration | `Model` / `PostProcess` / CMake |
| New **task** | Add a `PostProcess` subclass + result type | `Model` / `Decoder` |
| New **EP** | One case (no new class) | The entire inference pipeline |
| Custom **preprocessing** | `PreProcessParams::for_model()` | Decode / postprocess |

> Full steps, interface signatures, Decoder contract, and caveats — see **[docs/extending.md](docs/extending.md)**

## 📁 Project Layout

```
yolo-onnx/
├── include/yolo_onnx/            # Public API (only these two headers)
│   ├── yolo_onnx.hpp             #   Model / Config / inference entry points
│   └── yolo_onnx_types.hpp       #   Box / Mask / TensorSet / InferResult and other header-only types
├── src/
│   ├── core/                     # Inference backends
│   │   ├── backend.hpp               #   Abstract Backend interface
│   │   ├── backend_factory.cpp       #   create_backend() registration point
│   │   ├── onnxruntime_backend.*     #   Main line: EP dispatch
│   │   └── openvino_backend.*        #   Standalone backend (not compiled by default)
│   ├── process/
│   │   ├── preprocess/               #   letterbox + NCHW packing
│   │   └── postprocess/
│   │       ├── postprocess_core.hpp  #   Zero-dependency free functions (nms/iou/restore_*)
│   │       ├── detect/segment/pose/obb/sem.cpp
│   │       └── decoder/              #   Per-version decoders, organized by output format
│   ├── model.cpp                  # Single Model class (PIMPL)
│   └── model_impl.hpp
├── examples/                     # detect / segment / pose / obb / sem CLI examples
├── tools/                        # Accuracy-verification toolchain (see docs/accuracy.md)
├── tests/test_yolo.cpp           # 84 unit tests
├── cmake/FindONNXRuntime.cmake
├── docs/                         # extending.md · backends.md · accuracy.md · macos.md
└── assets/                       # Sample models and images
```

## 📚 Documentation

The main README covers usage only; more detailed reference docs live in their own places:

| Document | Contents |
|------|------|
| **[docs/accuracy.md](docs/accuracy.md)** | Accuracy-validation toolchain: dump_json / ref_onnx / compare / mask validation / preprocessing sweeps + historical bug post-mortems |
| **[docs/backends.md](docs/backends.md)** | Backends & EPs: full EP list, dedicated-API vs generic whitelist, standalone-runtime backends, pitfalls |
| **[docs/extending.md](docs/extending.md)** | Extension guide: full steps for all four axes, interface signatures, Decoder contract |
| **[docs/macos.md](docs/macos.md)** | macOS (Apple Silicon) deployment: dependencies, ONNX Runtime C SDK, OpenCV 4/5, CoreML EP, exFAT venv pitfall |
| **[python/README.md](python/README.md)** | Python bindings: build, usage, exported API |

## License

MIT
