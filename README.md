# yolo-onnx

基于 ONNX Runtime 的 C++ YOLO 推理框架，支持多种 YOLO 模型变体，设计为可扩展的多后端架构。

## 功能特性

### 支持的模型

| 模型 | 类型 | 输出格式 | 解码方式 |
|------|------|---------|---------|
| **YOLOv5** | Anchor-based | 3 输出 (P3/P4/P5) | sigmoid + exp + anchor |
| **YOLOX** | Anchor-free | 3 输出 / 1 拼接 | sigmoid + exp |
| **YOLOv8** | Anchor-free | 1 拼接输出 [1,C,N] | 直接解码 / DFL |
| **YOLOv11** | Anchor-free | 1 拼接输出 [1,C,N] | 同 v8 |
| **YOLO26** | Anchor-free | 1 拼接输出 [1,C,N] | 同 v8 |
| **PPYOLOE** | Anchor-free | 6 输出 (3cls+3reg) / 3 输出 / 1 拼接 | sigmoid + exp |

### 推理后端架构

```
┌─────────────────────────────────────────────────────────┐
│                     yolo_onnx                            │
│ ┌──────────┐  ┌──────────┐  ┌──────────────────┐        │
│ │ ModelV5  │  │ ModelV8  │  │ ModelPPYOLOE     │        │
│ │ ModelYOLOX│  │ (v11/v26)│  │                  │        │
│ └─────┬────┘  └────┬─────┘  └────────┬─────────┘        │
│       │             │                 │                  │
│ ┌─────┴─────────────┴─────────────────┴──────────────┐  │
│ │              Backend Interface                      │  │
│ │          load() / forward() / info()                │  │
│ └─────┬──────────────┬────────────────┬───────────────┘  │
│       │              │                │                  │
│ ┌─────┴─────┐  ┌─────┴─────┐  ┌──────┴───────┐          │
│ │ onnxruntime│  │  tensorrt  │  │  cann  │ rknn│          │
│ │   (CPU)   │  │ (GPU/FP16) │  │(Ascend│(Rockchip)      │
│ └───────────┘  └───────────┘  └────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 依赖

- C++17 编译器
- OpenCV (>= 4.x)
- ONNX Runtime (>= 1.12)

### 构建

```bash
# 安装依赖
# Ubuntu:
sudo apt install libopencv-dev

# 下载 ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.18.0

# 基本构建（仅 ONNX Runtime CPU 后端）
cd yolo-onnx
mkdir build && cd build
cmake .. -DWITH_EXAMPLES=ON -DWITH_TESTS=ON
make -j$(nproc)

# 构建时启用特定后端（需要预先安装对应 SDK）
# TensorRT 后端（NVIDIA GPU）
cmake .. -DWITH_EXAMPLES=ON -DWITH_TENSORRT=ON \
    -DTENSORRT_DIR=/path/to/TensorRT

# CANN 后端（Huawei Ascend NPU）
cmake .. -DWITH_EXAMPLES=ON -DWITH_CANN=ON \
    -DASCEND_DIR=/path/to/Ascend

# RKNN 后端（Rockchip NPU）
cmake .. -DWITH_EXAMPLES=ON -DWITH_RKNN=ON \
    -DRKNN_DIR=/path/to/rknn

# 启用多个后端
cmake .. -DWITH_EXAMPLES=ON \
    -DWITH_TENSORRT=ON -DWITH_CANN=ON -DWITH_RKNN=ON
```

### 使用示例

```bash
# 单图推理（ONNX Runtime CPU 后端）
./examples/detect_image v8 /path/to/yolov8n.onnx input.jpg output.jpg

# 使用 TensorRT 后端（NVIDIA GPU）
./examples/detect_image v8 model.onnx input.jpg output.jpg \
    --backend=tensorrt --device=0 --fp16

# 使用 CANN 后端（Ascend NPU，需要 .om 模型）
./examples/detect_image v8 model.om input.jpg output.jpg \
    --backend=cann --device=0

# 使用 RKNN 后端（Rockchip NPU，需要 .rknn 模型）
./examples/detect_image v8 model.rknn input.jpg output.jpg \
    --backend=rknn

# 自定义参数
./examples/detect_image v5 model.onnx input.jpg output.jpg \
    --score=0.5 --nms=0.45 --size=640 --classes=80 --threads=4
```

### 代码中使用

```cpp
#include "yolo_onnx/model.hpp"
#include <opencv2/opencv.hpp>

// 1. 创建模型
auto model = yolo_onnx::create_model(yolo_onnx::ModelType::YOLOv8);

// 2. 配置并加载
yolo_onnx::Model::Config config;
config.model_path   = "yolov8n.onnx";
config.score_thresh = 0.5f;
config.nms_thresh   = 0.45f;
config.num_classes  = 80;
model->load(config);

// 3. 推理
cv::Mat image = cv::imread("image.jpg");
auto boxes = model->infer(image);

// 4. 遍历结果
for (const auto& box : boxes) {
    printf("label=%d score=%.3f [%.0f,%.0f,%.0f,%.0f]\n",
           box.label, box.score, box.x1, box.y1, box.x2, box.y2);
}
```

## 扩展新后端

实现 `Backend` 接口并注册到工厂即可：

```cpp
#include "yolo_onnx/backend.hpp"

class MyBackend : public yolo_onnx::Backend {
    // 实现所有虚函数...
};

// 在 create_backend() 中注册
// 参见 src/backends/onnxruntime_backend.cpp
```

### 已支持的后端

| 后端 | 名称 | 目标平台 | 模型格式 | 精度支持 |
|------|------|---------|---------|---------|
| **ONNX Runtime** | `onnxruntime` | CPU | `.onnx` | FP32 |
| **TensorRT** | `tensorrt` | NVIDIA GPU | `.onnx` / `.engine` | FP32 / FP16 / INT8 |
| **CANN** | `cann` | Huawei Ascend NPU | `.om` | FP16 / INT8 |
| **RKNN** | `rknn` | Rockchip NPU | `.rknn` | FP16 / INT8 |

### 构建依赖

| 后端 | 依赖 | 环境变量 | CMake 选项 |
|------|------|---------|-----------|
| ONNX Runtime | onnxruntime | `ONNXRUNTIME_DIR` | 默认启用 |
| TensorRT | TensorRT + CUDA | `TENSORRT_DIR` | `-DWITH_TENSORRT=ON` |
| CANN | AscendCL | `ASCEND_DIR` | `-DWITH_CANN=ON` |
| RKNN | rknn_api | `RKNN_DIR` | `-DWITH_RKNN=ON` |

### 模型转换指南

```bash
# ONNX → TensorRT (自动在加载时构建，会自动缓存为 .engine)
./examples/detect_image v8 model.onnx input.jpg output.jpg --backend=tensorrt

# ONNX → CANN .om (使用 atc 工具)
atc --model=model.onnx --framework=5 --output=model \
    --soc_version=Ascend310P3 --input_format=NCHW

# ONNX → RKNN (使用 rknn-toolkit, Python)
# python -m rknn.api RKNN
# rknn.load_onnx(model.onnx)
# rknn.export_rknn(model.rknn)
```

## 扩展新模型

继承 `Model` 基类，实现 `decode_output()` 纯虚函数：

```cpp
class MyYOLO : public yolo_onnx::Model {
    BoxArray decode_output(
        const std::vector<std::vector<float>>&   output_data,
        const std::vector<std::vector<int64_t>>& output_shapes
    ) const override;
};
```

## Python 绑定（计划中）

```bash
cmake .. -DWITH_PYTHON=ON
```

```python
import yolo_onnx

model = yolo_onnx.Model("yolov8n.onnx", type="v8")
boxes = model.infer("image.jpg")
```

## 项目结构

```
yolo-onnx/
├── CMakeLists.txt
├── include/yolo_onnx/
│   ├── types.hpp           # 核心类型：Box, ModelType, 工具函数
│   ├── backend.hpp         # 推理后端接口
│   ├── model.hpp           # 模型接口 + 工厂
│   ├── backends/
│   │   ├── onnxruntime_backend.hpp
│   │   ├── tensorrt_backend.hpp   # TensorRT (GPU)
│   │   ├── cann_backend.hpp       # CANN (Ascend NPU)
│   │   └── rknn_backend.hpp       # RKNN (Rockchip NPU)
│   └── models/
│       ├── model_v5.hpp
│       ├── model_yolox.hpp
│       ├── model_v8.hpp
│       ├── model_v8_obb.hpp
│       ├── model_v8_pose.hpp
│       ├── model_v8_segment.hpp
│       └── model_ppyoloe.hpp
├── cmake/
│   ├── FindONNXRuntime.cmake
│   ├── FindTensorRT.cmake
│   ├── FindCANN.cmake
│   └── FindRKNN.cmake
├── src/
│   ├── backends/
│   │   ├── onnxruntime_backend.cpp
│   │   ├── tensorrt_backend.cpp
│   │   ├── cann_backend.cpp
│   │   └── rknn_backend.cpp
│   ├── model.cpp
│   └── models/
│       ├── model_v5.cpp
│       ├── model_yolox.cpp
│       ├── model_v8.cpp
│       ├── model_v8_obb.cpp
│       ├── model_v8_pose.cpp
│       ├── model_v8_segment.cpp
│       └── model_ppyoloe.cpp
├── examples/
│   ├── CMakeLists.txt
│   ├── detect_image.cpp
│   ├── obb_image.cpp
│   ├── pose_image.cpp
│   └── segment_image.cpp
├── tests/
│   ├── CMakeLists.txt
│   └── test_yolo.cpp
└── python/
    ├── CMakeLists.txt
    └── python_bindings.cpp
```

## License

MIT