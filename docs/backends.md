# 推理后端与 Execution Provider

推理引擎层（`src/core/`）的设计与使用。所有后端消费**同一份 `.onnx`**，转换与图优化在各自运行时内部完成，因此新增后端不需要改动上层的预处理 / 解码 / 后处理管线。

## 📑 目录

- [设计约定](#设计约定)
- [主线：ONNX Runtime + EP](#主线onnx-runtime--ep)
- [独立运行时后端](#独立运行时后端)
- [Backend 接口](#backend-接口)
- [新增一个 EP](#新增一个-ep)
- [踩过的坑](#踩过的坑)

## 设计约定

| 约定 | 说明 |
|------|------|
| 模型输入统一 | 全部后端吃 `.onnx`；厂商格式（`.om` / `.rknn`）由各自运行时在加载时转换 |
| 编译期零 EP 依赖 | EP 动态库（`libonnxruntime_providers_*.so`）由 onnxruntime 在运行时 `dlopen`，链接期不引入 |
| 换 EP 不重编译 | 通过 `Config::custom_config` 的 `ep=` 选择，同一个二进制直接跑 |
| 不可用则回退 | EP 注册失败（库缺失 / 不支持当前平台）时打印原因并回退 CPU，**不让整个推理失败** |
| 无格式知识 | 后端只负责 `load` / `forward` / 张量信息，不做任何解码 |

## 主线：ONNX Runtime + EP

硬件加速统一走 ONNX Runtime 的 Execution Provider：

```bash
./examples/detect_image v11 model.onnx bus.jpg out.jpg --ep=cuda --device=0
```

`ep=` 的可用值：

| EP | 走的 API | 说明 |
|----|---------|------|
| `cpu` | 默认，无需注册 | 缺省即 CPU |
| `cuda` | `AppendExecutionProvider_CUDA` | NVIDIA GPU，`--device=N` 选卡 |
| `tensorrt` | `AppendExecutionProvider_TensorRT` | NVIDIA GPU，`--fp16` 开半精度，带 engine 缓存 |
| `openvino` | `AppendExecutionProvider_OpenVINO` | Intel CPU/GPU，device_type 为 `CPU_FP32`/`GPU_FP16` 等 |
| `rocm` | `AppendExecutionProvider_ROCM` | AMD GPU |
| `migraphx` | `AppendExecutionProvider_MIGraphX` | NVIDIA MIGraphX |
| `vitisai` | `AppendExecutionProvider_VitisAI` | AMD VitisAI，额外 kv 原样透传 |
| `qnn` / `snpe` / `xnnpack` / `coreml` / `dml` / `webnn` / `webgpu` / `azure` / `js` / `nvtensorrtrtx` | 通用 `AppendExecutionProvider` | 白名单 EP，见下方"白名单"说明 |

配置项：

```bash
--ep=tensorrt --device=0 --fp16
```

对应 `Config::custom_config = "ep=tensorrt;--device=0;--fp16"`。除 `ep` / `device` / `--fp16` 外的 `key=value` 会**原样透传**给 EP 的 provider options。

## 独立运行时后端

不走 ORT、直接调厂商 SDK 的后端。代码保留但**默认不编译、不对外暴露**，需要时用编译宏打开——这样主线保持轻量，不绑架未使用的 SDK 依赖。

| 宏 | 后端 | 文件 | 说明 |
|----|------|------|------|
| `YOLO_ONNX_WITH_OPENVINO` | `openvino` | `openvino_backend.{hpp,cpp}` | 直接调 `ov::Core`，`device=CPU`/`GPU`/`AUTO` |
| `YOLO_ONNX_WITH_CANN` | *预留* | — | 昇腾，onnx → ATC → `.om` |
| `YOLO_ONNX_WITH_RKNN` | *预留* | — | 瑞芯微，onnx → rknn-toolkit2 → `.rknn` |

注册点在 [`backend_factory.cpp`](../src/core/backend_factory.cpp)，用 `#ifdef` 包起来：

```cpp
std::shared_ptr<Backend> create_backend(const std::string& backend_name) {
    if (backend_name == "onnxruntime" || backend_name.empty()) {
        return std::make_shared<OnnxruntimeBackend>();
    }
#ifdef YOLO_ONNX_WITH_OPENVINO
    if (backend_name == "openvino") {
        return std::make_shared<OpenvinoBackend>();
    }
#endif
    // 未知后端 → 报错并列出可用项
}
```

> **为什么 OpenVINO 需要独立后端**：官方 ORT 发行包**不含** `libonnxruntime_providers_openvino.so`（该 EP 只随 Intel 的自定义 ORT 构建发布）。所以要支持 OpenVINO，要么换 Intel 的 ORT，要么走这个独立后端。实测本机用 pip 版 OpenVINO 2026.4 走独立后端可正常推理（CPU 32.1 ms / 帧，iGPU 14.3 ms / 帧）。

启用方式：

```bash
# 需要先有 OpenVINO C++ SDK（pip install openvino 自带：headers + OpenVINOConfig.cmake）
cmake .. -DCMAKE_PREFIX_PATH=<site-packages>/openvino -DCMAKE_CXX_FLAGS="-DYOLO_ONNX_WITH_OPENVINO=1"
# 并把 openvino_backend.cpp 加入 YOLO_ONNX_SOURCES
```

## Backend 接口

```cpp
class Backend {
public:
    struct Config {
        std::string model_path;
        int         device_id    = 0;
        int         num_threads  = 4;
        bool        enable_fp16  = false;
        bool        enable_int8  = false;
        std::string custom_config;   // 后端私有配置（key=value;key=value）
    };

    virtual bool load(const Config& config) = 0;
    virtual bool forward(const std::vector<std::string>&    input_names,
                         const std::vector<std::vector<int64_t>>& input_shapes,
                         const std::vector<cv::Mat>&       input_data,
                         const std::vector<std::string>&    output_names,
                         TensorSet&                         outputs) = 0;
    virtual std::vector<std::string>          get_input_names()  const = 0;
    virtual std::vector<std::vector<int64_t>> get_input_shapes() const = 0;
    virtual std::vector<std::string>          get_output_names()  const = 0;
    virtual std::vector<std::vector<int64_t>> get_output_shapes() const = 0;
    virtual std::string name() const = 0;
};
```

约定：

- **输入**：每个输入一个连续 float32 的 `cv::Mat`（预处理产出 NCHW 4 维 Mat）
- **输出**：填 `TensorSet`（`vector<cv::Mat> datas` + `vector<vector<int64_t>> shapes`），每项是一个连续的 1-D float32 缓冲
- 后端不做解码、不关心语义，只搬运张量

## 新增一个 EP

**绝大多数情况不需要写代码**，两步即可：

1. 确认 EP 名字是否已在 `canonical_ep_name()` 表里（`onnxruntime_backend.cpp`）。名字是**大小写混合**的——`CoreML` / `DML` / `MIGraphX` / `VitisAI` / `NvTensorRtRtx`，所以需要一张规范化表，让用户写 `ep=coreml` / `ep=CoreML` 都能命中
2. 若是**专用 API** 的 EP（CUDA / TensorRT / ROCm / OpenVINO / MIGraphX / VitisAI），在 `setup_execution_providers()` 的分派 `if-else` 链里加一个 case

不需要新建类、不需要改 CMake、不需要重新编译框架。

只有当 EP 需要**专用 options 结构体**（而非简单 kv）时，才在该 case 里构造对应 options 并调用对应的 `AppendExecutionProvider_*`。

## 踩过的坑

### 通用 `AppendExecutionProvider(name, opts)` 只接受白名单

ORT ≥ 1.23 的 generic API **只接受**：QNN / SNPE / XNNPACK / OpenVINO / CoreML / DML / WEBNN / WebGPU / AZURE / JS / VitisAI / NvTensorRtRtx / MIGraphX。

传 CUDA 或 TensorRT 会报 `Unknown provider name 'CUDA'`。**这两个必须用专用 API**：

```cpp
session_options_.AppendExecutionProvider_CUDA(cuda_opts);      // ✅
session_options_.AppendExecutionProvider_TensorRT(trt_opts);  // ✅
session_options_.AppendExecutionProvider("CUDA", opts);       // ❌ Unknown provider name
```

### EP 名字大小写敏感，不能一律 toupper

`CoreML` 全大写成 `COREML` 就匹配不上（白名单里是混合大小写）。走 `canonical_ep_name()` 规范化，不要 `toupper` 用户输入。

### TensorRT 每次加载都重建 engine

`OrtTensorRTProviderOptions` 默认不缓存 engine，而 build 一次要几十秒（本机 yolo11n 约 54 秒）。已默认开启缓存：

```cpp
trt_opts.trt_engine_cache_enable = 1;
trt_opts.trt_engine_cache_path = "/tmp";
```

> engine 与 **GPU 架构绑定**，换卡必须清缓存（`rm /tmp/*.engine` / `rm /tmp/*trt*cache*`）。

另外 `trt_max_partition_iterations` / `trt_min_subgraph_size` 不设会触发 ORT 的告警日志，已显式设为 1000 / 1。

### CUDA EP 没有 fp16 开关

`OrtCUDAProviderOptionsV2` 在 1.23 是**不透明句柄**（只能 `Create*ProviderOptions` + `Update()` 配置），且它的 provider options 里**没有 `use_fp16` 键**——传了会报 `Unknown provider option: "use_fp16"`。

**fp16 只有 TensorRT EP 支持。** `--fp16` 对 CUDA EP 无效果。

### OpenVINO EP 的 device_id 是裸指针

`OrtOpenVINOProviderOptions::device_id` 是 `const char*`，指向的字符串必须覆盖整个 `AppendExecutionProvider_OpenVINO` 调用周期，不能传临时 `std::to_string(...)` 的结果。

### CoreML EP（macOS / Apple Silicon）

实测环境：Apple Silicon + macOS 27 + `onnxruntime-osx-arm64-1.30.0`，模型 `yolo11n@640`，输入 bus.jpg，**同一进程内 load 一次、infer 40 次**（计时不含 load）：

| 配置 | load | infer 中位数 | p90 |
|------|------|------------|-----|
| CPU（4 线程） | 25 ms | 24.6 ms | 28.0 ms |
| `ep=coreml`（默认 NeuralNetwork 格式） | **4039 ms** | 7.1 ms | 15.3 ms |
| `ep=coreml;ModelFormat=MLProgram;MLComputeUnits=CPUAndGPU` | 206 ms | **5.3 ms** | 6.1 ms |

精度与 CPU 一致：框最大偏差 1 px，score 差 ±0.005（GPU 走 fp16）。

**怎么确认真的跑在 CoreML 上**（而不是注册成功却回退了）：

```bash
ORT_LOGGING_LEVEL=3 ./build/examples/detect_image v11 model.onnx bus.jpg out.jpg --ep=coreml 2>&1 | grep -i coreml
# CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 5
# number of nodes in the graph: 320 number of nodes supported by CoreML: 315
```

**三个坑**：

1. **`device_id` 会被 CoreML 拒绝**。白名单分支过去无条件注入 `device_id`，CoreML 报 `Unknown option: device_id` → 注册失败 → **静默回退 CPU**（日志有，容易看漏）。现在只在显式指定非 0 设备时才透传。
2. **默认 NeuralNetwork 格式每次加载付 ~4 秒编译**，单次推理再快也救不回来（进程短、跑一次就退出的场景反而比 CPU 慢 40 倍）。换 `ModelFormat=MLProgram` 后 load 206 ms、infer 5.3 ms，两头都赢。1.30 **没有**缓存编译产物的 provider option（`ModelListDirectory` 已不被接受，会直接拒绝注册），所以别指望第二次进程变快。
3. **`MLComputeUnits` 的取值不带前缀**：`CPUOnly` / `CPUAndGPU` / `CPUAndNeuralEngine`。写成 `MLComputeUnitsCPUAndGPU` 会报 `Invalid value for option 'MLComputeUnits'` 并回退 CPU。

**例外：语义分割头不适合 CoreML。** `yolo26s-sem@1024` 在默认 NeuralNetwork 格式下**运行期直接失败**（`Unable to compute the prediction using a neural network model, error code: -1`，注意 load 阶段不报错），换 MLProgram 能出结果但更慢：CPU 121 ms / MLProgram+CPUAndGPU 165 ms / MLProgram 默认(ANE) 767 ms。detect / segment / pose 在 CoreML 下均正常。

> EP 私有选项（`ModelFormat` / `MLComputeUnits` 等）目前只能走 `Config::custom_config`——`examples/` 的命令行只解析 `--ep` / `--device` / `--fp16`。

### CANN 的 options 类型不完整

ORT 1.23 头文件里没有可用的 `OrtCANNProviderOptions` 定义（有 `AppendExecutionProvider_CANN` 声明但结构体不完整），因此暂未接线。