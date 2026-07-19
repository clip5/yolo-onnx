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
┌─────────────────────────────────────────────────┐
│                   yolo_onnx                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │
│  │  ModelV5  │  │  ModelV8  │  │  ModelPPYOLOE  │ │
│  │  ModelYOLOX│  │ (v11/v26) │  │                │ │
│  └─────┬─────┘  └────┬─────┘  └───────┬────────┘ │
│        │              │                │          │
│  ┌─────┴──────────────┴────────────────┴──────┐  │
│  │              Backend Interface              │  │
│  │          load() / forward() / info()        │  │
│  └─────┬──────────────┬────────────────┬───────┘  │
│        │              │                │          │
│  ┌─────┴─────┐  ┌─────┴─────┐  ┌──────┴───────┐  │
│  │ onnxruntime│  │  tensorrt  │  │  cann/npu   │  │
│  │   (CPU)   │  │  (future)  │  │  (future)   │  │
│  └───────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
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

# 构建
cd yolo-onnx
mkdir build && cd build
cmake .. -DWITH_EXAMPLES=ON -DWITH_TESTS=ON
make -j$(nproc)
```

### 使用示例

```bash
# 单图推理
./examples/detect_image v8 /path/to/yolov8n.onnx input.jpg output.jpg

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

class TensorRTBackend : public yolo_onnx::Backend {
    // 实现所有虚函数...
};

// 在 create_backend() 中注册
// src/backends/onnxruntime_backend.cpp
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
│   │   └── onnxruntime_backend.hpp
│   └── models/
│       ├── model_v5.hpp
│       ├── model_yolox.hpp
│       ├── model_v8.hpp
│       └── model_ppyoloe.hpp
├── src/
│   ├── cmake/FindONNXRuntime.cmake
│   ├── backends/onnxruntime_backend.cpp
│   ├── model.cpp
│   └── models/
│       ├── model_v5.cpp
│       ├── model_yolox.cpp
│       ├── model_v8.cpp
│       └── model_ppyoloe.cpp
├── examples/
│   └── detect_image.cpp
├── tests/
│   └── test_yolo.cpp
└── python/
    └── CMakeLists.txt
```

## License

MIT