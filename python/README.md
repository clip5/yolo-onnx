# Python 绑定

`yolo_onnx` 的 pybind11 绑定，把 C++ 推理管线（预处理 → ONNX Runtime → 解码 → NMS）暴露给 Python。

## 构建

需要 `pybind11`：

```bash
pip install "pybind11[global]"

cd build && cmake .. -DWITH_PYTHON=ON
make -j$(nproc)
```

产物：`build/python/yolo_onnx_py*.so`，可直接 `import`（或用 `PYTHONPATH=build/python`）。

> Python 绑定**只编译 ORT 后端**，与主库一致。EP 的完整说明见 [src/core/README.md](../src/core/README.md)（注意 `ModelConfig` 暂未暴露 `custom_config`，见下文）。

## 快速上手

最简形式——一行完成加载 + 推理：

```python
import cv2
import yolo_onnx_py as yolo

img = cv2.imread("bus.jpg")
result = yolo.detect("yolo11n.onnx", img, model_type_str="v11")

for box in result.boxes:
    print(f"label={box.label} score={box.score:.3f} "
          f"rect=[{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}]")
```

也可以从文件读图（少一次 imread）：

```python
result = yolo.detect_file("yolo11n.onnx", "bus.jpg", model_type_str="v11")
```

## 完整控制：Model 对象

需要精细控制时用 `Model`：

```python
import cv2
import yolo_onnx_py as yolo

model = yolo.create_model(yolo.ModelType.YOLOv11, yolo.TaskType.Detect)

cfg = yolo.ModelConfig()
cfg.model_path   = "yolo11n.onnx"
cfg.score_thresh = 0.35
cfg.nms_thresh   = 0.45
cfg.num_threads  = 8
model.load(cfg)

result = model.infer(cv2.imread("bus.jpg"))     # 或 model.infer_file("bus.jpg")
```

> `ModelConfig` 目前只暴露了模型/阈值/尺寸/线程等字段，**尚未暴露 `custom_config`**，因此 Python 侧暂时只能用默认 CPU 后端。换 EP（`ep=cuda` 等）需要在 C++ 侧配置。

`Model` 上的推理入口：

| 方法 | 返回 |
|------|------|
| `infer(img)` / `infer_file(path)` | `InferResult`（按 `TaskType` 返回对应结果对象） |
| `infer_detect(img)` | `DetectResult` |
| `infer_segment(img)` / `infer_segment_file(path)` | `SegmentResult` |
| `infer_pose(img)` / `infer_pose_file(path)` | `PoseResult` |
| `infer_obb(img)` / `infer_obb_file(path)` | `OBBResult` |

分割示例：

```python
model = yolo.create_model(yolo.ModelType.YOLOv11, yolo.TaskType.Segment)
cfg = yolo.ModelConfig()
cfg.model_path = "yolo11s-seg640.onnx"
model.load(cfg)

seg = model.infer_segment_file("bus.jpg")
for box, mask in zip(seg.boxes, seg.masks):
    print(box.label, box.score, mask.data.shape)   # mask 已还原到原图尺寸
```

## 导出的 API

**枚举**：`ModelType`（`YOLOv5` / `YOLOX` / `YOLOv8` / `YOLOv11` / `YOLO26` / `PPYOLOE`）、`TaskType`（`Detect` / `Segment` / `Pose` / `OBB`）

**结果类型**：`Box` / `Keypoint` / `Mask` / `OBBBox` / `DetectResult` / `SegmentResult` / `PoseResult` / `OBBResult` / `LetterboxInfo`

**函数**：
- `create_model(model_type)` / `create_model(model_type, task)`
- `detect(...)` / `detect_file(...)` —— 便捷入口
- `model_type_name(t)` / `task_type_name(t)`
- `iou(a, b)` / `nms(boxes, iou_thresh)` / `nms_class_aware(boxes, iou_thresh)`
- `obb_iou(a, b)` / `obb_nms(boxes, iou_thresh)` / `sigmoid(x)`

后处理原语（`iou` / `nms` / `obb_nms`）直接暴露出来了——可以在 Python 侧做二次筛选或自定义 NMS。

## 说明

- 输入图像是 **BGR** 的 `cv2.UMat` / numpy 数组（`uint8`，HWC）；框架内部负责 letterbox 与通道序转换
- C++ 侧的风格是"返回强类型结果"，所以这里没有 `std::variant` 的对应物——`TaskType` 决定了返回哪个结果类
- 详细字段见 C++ 头文件 [`include/yolo_onnx/yolo_onnx_types.hpp`](../include/yolo_onnx/yolo_onnx_types.hpp)（所有类型都是 header-only 的）