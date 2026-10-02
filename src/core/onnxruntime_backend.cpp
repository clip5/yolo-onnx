#include "core/onnxruntime_backend.hpp"
#include <iostream>
#include <cstring>
#include <sstream>

namespace yolo_onnx {

namespace {

/// ONNX Runtime 的 EP 名字是大小写混合的（CoreML / DML / QNN / MIGraphX /
/// NvTensorRtRtx …），generic AppendExecutionProvider 按字符串精确匹配，
/// 所以不能一律 toupper。这里做一次规范化，让用户写 ep=coreml /
/// ep=CoreML / ep=COREML 都能命中。
std::string canonical_ep_name(const std::string& raw) {
    static const std::unordered_map<std::string, std::string> kNames = {
        {"CPU",         "CPU"},
        {"CUDA",        "CUDA"},
        {"TENSORRT",    "TENSORRT"},
        {"TRT",         "TENSORRT"},
        {"OPENVINO",    "OPENVINO"},
        {"ROCM",        "ROCM"},
        {"MIGRAPHX",    "MIGraphX"},
        {"VITISAI",     "VitisAI"},
        {"QNN",         "QNN"},
        {"SNPE",        "SNPE"},
        {"XNNPACK",     "XNNPACK"},
        {"COREML",      "CoreML"},
        {"DML",         "DML"},
        {"WEBNN",       "WEBNN"},
        {"WEBGPU",      "WebGPU"},
        {"AZURE",       "AZURE"},
        {"JS",          "JS"},
        {"NVTENSORRTRTX", "NvTensorRtRtx"},
    };

    std::string upper;
    upper.reserve(raw.size());
    for (auto c : raw) upper += static_cast<char>(toupper(static_cast<unsigned char>(c)));

    auto it = kNames.find(upper);
    if (it != kNames.end()) return it->second;

    // 未知 EP：原样交给 ORT，让它自己报错（错误信息更准确）
    return upper;
}

} // namespace

OnnxruntimeBackend::OnnxruntimeBackend()
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , session_(nullptr)
{
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OnnxruntimeBackend::~OnnxruntimeBackend() = default;

bool OnnxruntimeBackend::setup_execution_providers(const std::string& custom_config,
                                                   int device_id,
                                                   int num_threads) {
    // Parse custom_config for ep=xxx
    // Format: key=value;key=value;...
    // Supported keys:
    //   ep      — execution provider name (cuda, tensorrt, cpu, ...)
    //   device  — device ID (overrides Config::device_id for EP)

    if (custom_config.empty()) {
        // Default: CPU
        name_ = "onnxruntime";
        return true;
    }

    std::string ep_name;
    int ep_device_id = device_id;

    // Parse key=value pairs
    bool enable_fp16_flag = false;
    std::unordered_map<std::string, std::string> ep_options;
    std::istringstream stream(custom_config);
    std::string token;
    while (std::getline(stream, token, ';')) {
        auto eq = token.find('=');
        if (eq == std::string::npos) {
            // 无值 flag（如 --fp16）
            std::string flag = token;
            auto trim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
                s.erase(s.find_last_not_of(" \t") + 1);
            };
            trim(flag);
            if (flag == "--fp16") enable_fp16_flag = true;
            continue;
        }

        std::string key = token.substr(0, eq);
        std::string val = token.substr(eq + 1);

        // Trim whitespace
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(val);

        if (key == "ep") {
            ep_name = canonical_ep_name(val);
        } else if (key == "device" || key == "--device") {
            ep_device_id = std::stoi(val);
        } else {
            // 其余 key=value 原样透传给 EP（如 OpenVINO 的 device_type）
            ep_options[key] = val;
        }
    }

    if (ep_name.empty() || ep_name == "CPU") {
        // CPU is default, nothing to register
        name_ = "onnxruntime";
        return true;
    }

    // ============================================================
    // EP 分派
    // ============================================================
    // ONNX Runtime 1.23 里 generic 的 AppendExecutionProvider(name, opts)
    // 只接受固定白名单（QNN/SNPE/XNNPACK/OpenVINO/CoreML/...），
    // CUDA / TensorRT / ROCm / CANN 等必须走各自的专用 API。
    // provider 动态库（libonnxruntime_providers_*.so）由 onnxruntime
    // 在运行时 dlopen，编译期无需链接。
    //
    // 新增 EP 时只需在下面加一个分支；EP 不可用时 ORT 抛异常，
    // 这里统一回退 CPU，不让整个推理失败。
    // ============================================================
    try {
        if (ep_name == "CUDA") {
            // 注意：CUDA EP 没有 fp16 开关（V2 的 provider options 里也没有），
            // fp16 只有 TensorRT EP 支持。--fp16 对 CUDA EP 无效果。
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = ep_device_id;
            session_options_.AppendExecutionProvider_CUDA(cuda_opts);
        } else if (ep_name == "TENSORRT") {
            OrtTensorRTProviderOptions trt_opts{};
            trt_opts.device_id = ep_device_id;
            trt_opts.trt_fp16_enable = enable_fp16_flag;
            trt_opts.trt_max_partition_iterations = 1000;
            trt_opts.trt_min_subgraph_size = 1;
            // 缓存 engine，避免每次加载都重新 build（build 一次要几十秒）
            trt_opts.trt_engine_cache_enable = 1;
            trt_opts.trt_engine_cache_path = "/tmp";
            session_options_.AppendExecutionProvider_TensorRT(trt_opts);
        } else if (ep_name == "OPENVINO") {
            // OpenVINO EP：device_type 为 CPU_FP32/CPU_FP16/GPU_FP32/GPU_FP16
            OrtOpenVINOProviderOptions ov_opts{};
            // device_id 是 const char*，必须保证生命周期覆盖整个 Append 调用
            std::string device_id_str = std::to_string(ep_device_id);
            const char* device_type = nullptr;
            if (enable_fp16_flag) {
                device_type = ep_device_id == 0 ? "CPU_FP16" : "GPU_FP16";
            } else {
                device_type = ep_device_id == 0 ? "CPU_FP32" : "GPU_FP32";
            }
            ov_opts.device_type   = device_type;
            ov_opts.device_id     = ep_device_id == 0 ? nullptr : device_id_str.c_str();
            ov_opts.num_of_threads = num_threads;
            session_options_.AppendExecutionProvider_OpenVINO(ov_opts);
        } else if (ep_name == "ROCM") {
            OrtROCMProviderOptions rocm_opts{};
            rocm_opts.device_id = ep_device_id;
            session_options_.AppendExecutionProvider_ROCM(rocm_opts);
        } else if (ep_name == "MIGraphX") {
            OrtMIGraphXProviderOptions migraphx_opts{};
            migraphx_opts.device_id = ep_device_id;
            session_options_.AppendExecutionProvider_MIGraphX(migraphx_opts);
        } else if (ep_name == "VitisAI") {
            session_options_.AppendExecutionProvider_VitisAI(ep_options);
        } else {
            // 白名单 EP：QNN / SNPE / XNNPACK / CoreML / DML / WEBNN /
            // WebGPU / AZURE / JS / NvTensorRtRtx（名字已规范化）
            ep_options["device_id"] = std::to_string(ep_device_id);
            session_options_.AppendExecutionProvider(ep_name, ep_options);
        }
        name_ = "onnxruntime/" + ep_name;
        std::cout << "[OnnxruntimeBackend] Using execution provider: " << ep_name
                  << " (device=" << ep_device_id << ")" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[OnnxruntimeBackend] Failed to register EP '" << ep_name
                  << "': " << e.what() << std::endl;
        std::cerr << "[OnnxruntimeBackend] Falling back to CPU (make sure you have the "
                  << "correct onnxruntime package, e.g. onnxruntime-gpu for CUDA)" << std::endl;
        // Fall back to CPU — don't fail
    }

    return true;
}

bool OnnxruntimeBackend::load(const Config& config) {
    try {
        // Set threads
        session_options_.SetIntraOpNumThreads(config.num_threads);
        session_options_.SetInterOpNumThreads(config.num_threads);

        // Register execution provider (before creating session)
        setup_execution_providers(config.custom_config, config.device_id, config.num_threads);

        // Create session
        session_ = Ort::Session(env_, config.model_path.c_str(), session_options_);

        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_.GetInputCount();
        input_names_.resize(num_inputs);
        input_shapes_.resize(num_inputs);
        input_types_.resize(num_inputs);

        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session_.GetInputNameAllocated(i, allocator);
            input_names_[i] = name.get();

            auto type_info = session_.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_[i] = tensor_info.GetShape();
            input_types_[i] = tensor_info.GetElementType();
        }

        // Get output info
        size_t num_outputs = session_.GetOutputCount();
        output_names_.resize(num_outputs);
        output_shapes_.resize(num_outputs);
        output_types_.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session_.GetOutputNameAllocated(i, allocator);
            output_names_[i] = name.get();

            auto type_info = session_.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_[i] = tensor_info.GetShape();
            output_types_[i] = tensor_info.GetElementType();
        }

        loaded_ = true;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[OnnxruntimeBackend] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxruntimeBackend::forward(
    const std::vector<std::string>&          input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<cv::Mat>&             input_data,
    const std::vector<std::string>&          output_names,
    TensorSet&                               outputs
) {
    if (!loaded_) {
        std::cerr << "[OnnxruntimeBackend] Model not loaded" << std::endl;
        return false;
    }

    try {
        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names_cstr;

        for (size_t i = 0; i < input_names.size(); i++) {
            input_names_cstr.push_back(input_names[i].c_str());

            if (i >= input_data.size() || input_data[i].empty()) {
                std::cerr << "[OnnxruntimeBackend] Missing input tensor data at index "
                          << i << std::endl;
                return false;
            }

            // 每个输入对应一个连续 float32 的 Mat（PreProcess 产出NCHW cv::Mat）
            const cv::Mat& m = input_data[i];
            size_t total_elements = static_cast<size_t>(m.total() * m.channels());

            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                const_cast<float*>(m.ptr<float>()),
                total_elements,
                input_shapes[i].data(),
                input_shapes[i].size()
            ));
        }

        // Prepare output names
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names) {
            output_names_cstr.push_back(name.c_str());
        }

        // Run inference
        auto ort_outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(), input_tensors.data(), input_names.size(),
            output_names_cstr.data(), output_names.size()
        );

        // 拷贝输出到 TensorSet（每个输出一个连续 float32 Mat + 形状）
        outputs.datas.clear();
        outputs.shapes.clear();
        outputs.datas.resize(ort_outputs.size());
        outputs.shapes.resize(ort_outputs.size());

        for (size_t i = 0; i < ort_outputs.size(); i++) {
            auto tensor_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
            outputs.shapes[i] = tensor_info.GetShape();

            const size_t total_elements = tensor_info.GetElementCount();
            cv::Mat dst(1, (int)total_elements, CV_32F);
            float* out_ptr = dst.ptr<float>();

            // TensorSet 统一以 float32 承载。对整型输出（如语义分割头
            // Resize→ArgMax→Cast 后的 uint8/int32 类别图）逐元素转换即可——
            // 类别 id 远小于 2^24，float32 可无损表示。
            // 旧实现在这里直接拒绝一切非 float 输出，导致 yolo26s-sem
            // 报 "Unsupported output type: 2"(uint8) 而无法推理。
            switch (tensor_info.GetElementType()) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    std::memcpy(out_ptr, ort_outputs[i].GetTensorData<float>(),
                                total_elements * sizeof(float));
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                    const uint8_t* src = ort_outputs[i].GetTensorData<uint8_t>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = (float)src[k];
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                    const int8_t* src = ort_outputs[i].GetTensorData<int8_t>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = (float)src[k];
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
                    const uint16_t* src = ort_outputs[i].GetTensorData<uint16_t>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = (float)src[k];
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
                    const int16_t* src = ort_outputs[i].GetTensorData<int16_t>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = (float)src[k];
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    const int32_t* src = ort_outputs[i].GetTensorData<int32_t>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = (float)src[k];
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    const int64_t* src = ort_outputs[i].GetTensorData<int64_t>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = (float)src[k];
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                    const bool* src = ort_outputs[i].GetTensorData<bool>();
                    for (size_t k = 0; k < total_elements; k++) out_ptr[k] = src[k] ? 1.0f : 0.0f;
                    break;
                }
                default: {
                    std::cerr << "[OnnxruntimeBackend] Unsupported output type: "
                              << tensor_info.GetElementType() << std::endl;
                    return false;
                }
            }

            outputs.datas[i] = std::move(dst);
        }

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[OnnxruntimeBackend] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> OnnxruntimeBackend::get_input_names() const {
    return input_names_;
}

std::vector<std::vector<int64_t>> OnnxruntimeBackend::get_input_shapes() const {
    return input_shapes_;
}

std::vector<std::string> OnnxruntimeBackend::get_output_names() const {
    return output_names_;
}

std::vector<std::vector<int64_t>> OnnxruntimeBackend::get_output_shapes() const {
    return output_shapes_;
}

} // namespace yolo_onnx