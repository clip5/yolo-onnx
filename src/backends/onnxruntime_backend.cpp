#include "yolo_onnx/backends/onnxruntime_backend.hpp"
#include <iostream>
#include <cstring>
#include <sstream>

namespace yolo_onnx {

OnnxruntimeBackend::OnnxruntimeBackend()
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , session_(nullptr)
{
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OnnxruntimeBackend::~OnnxruntimeBackend() = default;

bool OnnxruntimeBackend::setup_execution_providers(const std::string& custom_config, int device_id) {
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
    std::istringstream stream(custom_config);
    std::string token;
    while (std::getline(stream, token, ';')) {
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;

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
            ep_name = val;
            // Uppercase the EP name for ONNX Runtime
            for (auto& c : ep_name) c = toupper(c);
        } else if (key == "device") {
            ep_device_id = std::stoi(val);
        }
    }

    if (ep_name.empty() || ep_name == "CPU") {
        // CPU is default, nothing to register
        name_ = "onnxruntime";
        return true;
    }

    // Register the execution provider (ONNX Runtime 1.14+ string-based API)
    // This does NOT require compile-time linking to EP-specific libraries —
    // the onnxruntime shared library handles it internally.
    std::unordered_map<std::string, std::string> ep_options;
    ep_options["device_id"] = std::to_string(ep_device_id);

    try {
        session_options_.AppendExecutionProvider(ep_name, ep_options);
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
        setup_execution_providers(config.custom_config, config.device_id);

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
    const std::vector<float>&                input_data,
    const std::vector<std::string>&          output_names,
    std::vector<std::vector<int64_t>>&       output_shapes,
    std::vector<std::vector<float>>&         output_data
) {
    if (!loaded_) {
        std::cerr << "[OnnxruntimeBackend] Model not loaded" << std::endl;
        return false;
    }

    try {
        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names_cstr;
        size_t data_offset = 0;

        for (size_t i = 0; i < input_names.size(); i++) {
            input_names_cstr.push_back(input_names[i].c_str());

            // Calculate total elements
            int64_t total_elements = 1;
            for (auto dim : input_shapes[i]) {
                total_elements *= dim;
            }

            // Create tensor from existing data
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                const_cast<float*>(input_data.data() + data_offset),
                total_elements,
                input_shapes[i].data(),
                input_shapes[i].size()
            ));

            data_offset += total_elements;
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

        // Copy output data
        output_data.resize(ort_outputs.size());
        output_shapes.resize(ort_outputs.size());

        for (size_t i = 0; i < ort_outputs.size(); i++) {
            auto tensor_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
            output_shapes[i] = tensor_info.GetShape();

            size_t total_elements = tensor_info.GetElementCount();
            output_data[i].resize(total_elements);

            // Get element type and copy accordingly
            auto element_type = tensor_info.GetElementType();
            if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                const float* data = ort_outputs[i].GetTensorData<float>();
                std::memcpy(output_data[i].data(), data, total_elements * sizeof(float));
            } else {
                std::cerr << "[OnnxruntimeBackend] Unsupported output type: " 
                          << element_type << std::endl;
                return false;
            }
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

// Backend factory — only ONNX Runtime backed by EP
std::shared_ptr<Backend> create_backend(const std::string& backend_name) {
    if (backend_name == "onnxruntime") {
        return std::make_shared<OnnxruntimeBackend>();
    }
    std::cerr << "[create_backend] Unknown backend: " << backend_name << std::endl;
    std::cerr << "[create_backend] Supported backends: onnxruntime" << std::endl;
    return nullptr;
}

} // namespace yolo_onnx