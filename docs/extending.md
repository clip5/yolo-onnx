# 扩展指南

框架有三个**正交**的扩展轴，各自独立、互不干扰：

| 想加什么 | 要改的地方 | 不用改的地方 |
|---------|-----------|-------------|
| 新的**导出格式**（某个 YOLO 变体头结构不同） | 加一个 `Decoder` 子类 | `Model` / `PostProcess` / CMake |
| 新的**任务**（如分类） | 加一个 `PostProcess` 子类 + 结果类型 | `Model` / `Decoder` |
| 新的**后端 / EP** | EP → 一个 case；独立运行时 → 一个 `Backend` 子类 | 整条推理管线 |

三条轴的交汇点只有两处工厂函数：`create_decoder()` 和 `create_postprocess()`。

```
image → PreProcess → Backend(forward) → TensorSet → Decoder → PostProcess → InferResult
        ①预处理        ③后端            统一张量       ②解码      任务后处理      结果
```

- ① 差异化预处理 → [`PreProcessParams::for_model()`](../src/process/preprocess/preprocess.cpp)
- ② 按输出格式解码 → [`create_decoder()`](../src/process/postprocess/decoder/decoder_factory.cpp)
- ③ 按任务分派 → [`create_postprocess()`](../src/process/postprocess/postprocess_factory.cpp)

## 📑 目录

- [轴一：新的导出格式](#轴一新的导出格式)
- [轴二：新的任务](#轴二新的任务)
- [轴三：新的后端 / EP](#轴三新的后端--ep)
- [轴四：差异化的预处理](#轴四差异化的预处理)

## 轴一：新的导出格式

**场景**：某个 YOLO 变体的输出头布局和现有的都不同，需要一个新的解码逻辑。

**要做的**（4 步，不涉及 `Model` 或 `PostProcess`）：

1. 在 `src/process/postprocess/decoder/decoder_<name>.cpp` 实现子类，覆写需要的 `decode_*`：

```cpp
#include "process/postprocess/decoder/decoder.hpp"

namespace yolo_onnx {

class MyVariantDecoder : public Decoder {
public:
    std::string name() const override { return "myvariant"; }

    // 按需覆写：不实现则返回空
    BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const override;
};

} // namespace yolo_onnx
```

2. 在 [`decoder.hpp`](../src/process/postprocess/decoder/decoder.hpp) 声明该类
3. 在 [`create_decoder()`](../src/process/postprocess/decoder/decoder_factory.cpp) 注册
4. 把新源文件加进 **`CMakeLists.txt` 和 `python/CMakeLists.txt` 两处**的源文件列表

```cpp
// decoder_factory.cpp
case ModelType::MyVariant: return std::make_shared<MyVariantDecoder>();
```

如果这个格式还需要不同的预处理，见[轴四](#轴四差异化的预处理)。

### Decoder 契约

```cpp
virtual BoxArray decode_detect(const TensorSet& out, const DecodeContext& ctx) const;
virtual std::vector<Candidate<std::array<float, 32>>> decode_segment(...) const;
virtual std::vector<Candidate<std::vector<Keypoint>>>  decode_pose(...) const;
virtual OBBBoxArray decode_obb(const TensorSet& out, const DecodeContext& ctx) const;
```

- 输入是原始张量，输出是**模型输入坐标系**下的候选框（不要在这里还原到原图尺寸，那是后处理的事）
- 网格尺寸**一律**从 `DecodeContext` 推导，**禁止**硬编码：

```cpp
ctx.grid_w(level)     // input_width  / stride(level)
ctx.grid_h(level)     // input_height / stride(level)
ctx.grid_count(level) // 两者之积
ctx.stride(level)     // 8 << level，即 8/16/32
```

> 历史上这里硬编码过 `80/40/20` 和 `6400/1600/400`，遇到非正方形输入（如 640×352）会解错网格并**读越界**。

### 别重复实现已有的工具

复用 [`postprocess_core.hpp`](../src/process/postprocess/postprocess_core.hpp) 里的自由函数（零依赖，可直接拷走）：

```cpp
iou / nms / nms_class_aware / obb_iou / obb_nms    // 图元
restore_*                                          // 坐标还原
filter_by_score / top_k                            // 筛选
detect_pipeline / obb_pipeline / select_indices    // 一步到位组合
```

## 轴二：新的任务

**场景**：加一个现有五任务之外的任务，比如分类。

1. 在 `src/process/postprocess/<task>.cpp` 实现 `PostProcess` 子类：

```cpp
class PostProcessClassify : public PostProcess {
public:
    using PostProcess::PostProcess;   // (params, decoder)
    InferResult forward(const TensorSet& outputs,
                        const LetterboxInfo& lb) const override;
};
```

2. 在 [`postprocess.hpp`](../src/process/postprocess/postprocess.hpp) 声明
3. 在 [`create_postprocess()`](../src/process/postprocess/postprocess_factory.cpp) 注册
4. 在 `include/yolo_onnx/yolo_onnx_types.hpp` 的 `InferResult` variant 里加结果类型：

```cpp
using InferResult = std::variant<DetectResult, SegmentResult,
                                 PoseResult, OBBResult, SemResult,
                                 ClassifyResult>;
```

5. 想让调用方拿到非 variant 的类型，加一个类型化入口：

```cpp
ClassifyResult Model::infer_classify(const cv::Mat& image);
```

后处理子类**只做任务分派**，具体逻辑复用 `postprocess_core.hpp`，不要重复实现 NMS / 坐标还原。

> **稠密任务（语义分割）是个例外**：它没有候选框，因此不经过 Decoder，也没有
> NMS / 置信度过滤 / `top_k`——`PostProcessSem` 只做「按 letterbox 裁 padding +
> 最近邻缩放到原图」两步。类别 id **必须**用最近邻缩放，双线性会在类别边界
> 插出根本不存在的 id（见 [`sem.cpp`](../src/process/postprocess/sem.cpp)）。

## 轴三：新的后端 / EP

分两种情况，详见 [`docs/backends.md`](backends.md)。

### (a) 能表达成 ONNX Runtime 的 EP（主流情况）

**不需要写类**，在 [`setup_execution_providers()`](../src/core/onnxruntime_backend.cpp) 的分派链里加一个 case：

```cpp
} else if (ep_name == "MYEP") {
    MyProviderOptions opts{};                       // 若需要专用 options
    session_options_.AppendExecutionProvider_MyEP(opts);
} else if (ep_name == "SomeWhitelistedEP") {
    ep_options["device_id"] = std::to_string(ep_device_id);
    session_options_.AppendExecutionProvider(ep_name, ep_options);
}
```

- EP 名字不在 `canonical_ep_name()` 表里的话，往表里加一条（名字大小写混合，不能一律 `toupper`）
- 忘了加表的话，用户写小写就匹配不上
- 编译期零依赖：EP 动态库由 onnxruntime 运行时 `dlopen`

### (b) 独立运行时（自己的 SDK，无法表达为 ORT EP）

1. 实现 `Backend`（`load` / `forward` / 四个张量信息方法 / `name`）到 `src/core/<name>_backend.{hpp,cpp}`
2. 在 [`create_backend()`](../src/core/backend_factory.cpp) 注册，用编译宏包起来，默认不暴露
3. 把源文件加进 `CMakeLists.txt` 与 `python/CMakeLists.txt`

已有一个范例：`openvino_backend.{hpp,cpp}`（由 `YOLO_ONNX_WITH_OPENVINO` 门控）。

所有后端消费**同一份 `.onnx`**，转换在各自运行时内部完成。后端不做解码，只搬运张量：输入是每输入一个连续 float32 `cv::Mat`，输出填 `TensorSet`。

## 轴四：差异化的预处理

**场景**：新格式需要不同的通道序 / 归一化 / padding 对齐。

改 [`PreProcessParams::for_model()`](../src/process/preprocess/preprocess.cpp)：

```cpp
PreProcessParams PreProcessParams::for_model(ModelType type, int width, int height) {
    PreProcessParams p;
    switch (type) {
        case ModelType::MyVariant:
            p.swap_rb      = false;   // 保持 BGR
            p.scale_factor = 1.0f;    // 不除 255
            p.align        = PadAlign::TopLeft;
            break;
    }
    return p;
}
```

可调项：

| 字段 | 含义 |
|------|------|
| `swap_rb` | BGR→RGB（保持 BGR 的模型设 `false`） |
| `scale_factor` | 像素缩放（`1/255` 是常规值；保留 0-255 原值则设 `1.0f`） |
| `mean` / `std` | 每通道归一化，**在 scale 之后**应用：`(v*scale - mean) / std` |
| `align` | padding 对齐：`Center` / `TopLeft` 等 |

> **模型检出率明显偏低时，先怀疑预处理，再怀疑权重。** 用 [`tools/yolox_pp_sweep.py`](accuracy.md#yolox_pp_sweeppy) 扫描「对齐方式 × 通道序 × 归一化」组合，拿一个已知表现良好的检测器当基准打分。YOLOX 就是这么定位的：官方预处理是 top-left 对齐 + 保留 0-255 + 无归一化，用默认值会让 bus.jpg 的检出从 5/5 掉到 2/5。

## 改完怎么验证

```bash
cd build && make -j$(nproc) && ./tests/test_yolo     # 回归测试
```

新增的导出格式建议同时加一条**结构性**回归测试——用构造的张量断言解码结果，而不是依赖某个权重文件。同时用 [`tools/`](accuracy.md) 的比对链路确认精度：

```bash
./tools/dump_json <type> detect model.onnx bus.jpg out/cpp/bus.json
python3 tools/ref_onnx.py --model model.onnx --task detect --image bus.jpg --out out/ref/bus.json
python3 tools/compare.py --cpp out/cpp/bus.json --ref out/ref/bus.json --iou 0.5
```