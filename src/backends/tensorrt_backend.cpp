#include "yolo_onnx/backends/tensorrt_backend.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>

namespace yolo_onnx {

// ============================================================
// TensorRT Logger
// ============================================================
void TensorRTBackend::TRTLogger::log(Severity severity, const char* msg) noexcept {
    // Suppress verbose warnings
    if (severity == Severity::kVERBOSE || severity == Severity::kINFO) return;
    switch (severity) {
        case Severity::kWARNING: std::cerr << "[TRT WARNING] " << msg << std::endl; break;
        case Severity::kERROR:   std::cerr << "[TRT ERROR] "   << msg << std::endl; break;
        case Severity::kINTERNAL_ERROR: std::cerr << "[TRT FATAL] " << msg << std::endl; break;
        default: break;
    }
}

// ============================================================
// TensorRTBackend
// ============================================================
TensorRTBackend::TensorRTBackend() = default;
TensorRTBackend::~TensorRTBackend() {
    // Free device memory
    for (auto& buf : device_buffers_) {
        if (buf) { cudaFree(buf); buf = nullptr; }
    }
    device_output_ptrs_.clear();
}

bool TensorRTBackend::load(const Config& config) {
    device_id_ = config.device_id;

    // Set device
    cudaSetDevice(device_id_);

    // Check file extension
    std::string ext;
    size_t dot_pos = config.model_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = config.model_path.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    bool success = false;
    if (ext == ".engine" || ext == ".plan") {
        success = load_engine(config.model_path);
    } else if (ext == ".onnx") {
        success = build_from_onnx(config.model_path, config.enable_fp16, config.enable_int8);
    } else {
        std::cerr << "[TensorRTBackend] Unsupported model format: " << ext << std::endl;
        return false;
    }

    if (!success) {
        return false;
    }

    // Create execution context
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        [](nvinfer1::IExecutionContext* ctx) { if (ctx) ctx->destroy(); }
    );
    if (!context_) {
        std::cerr << "[TensorRTBackend] Failed to create execution context" << std::endl;
        return false;
    }

    // Query I/O names and shapes
    int num_inputs  = engine_->getNbIOTensors();
    for (int i = 0; i < num_inputs; i++) {
        auto name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);

        std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);
        // Replace dynamic dimensions (-1) with a reasonable default
        for (auto& d : shape) {
            if (d == -1) d = 1;
        }

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_names_.push_back(name);
            input_shapes_.push_back(shape);
        } else {
            output_names_.push_back(name);
            output_shapes_.push_back(shape);
            output_sizes_.push_back(
                std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>())
            );
        }
    }

    // Allocate device memory
    // For simplicity, use a single input and up to N outputs
    // Input buffer
    size_t input_size = 1;
    for (auto& d : input_shapes_[0]) input_size *= d;
    cudaMalloc(&device_buffers_[0], input_size * sizeof(float));

    // Output buffers
    device_output_ptrs_.resize(output_names_.size());
    for (size_t i = 0; i < output_names_.size(); i++) {
        cudaMalloc(&device_output_ptrs_[i], output_sizes_[i] * sizeof(float));
    }

    loaded_ = true;
    return true;
}

bool TensorRTBackend::build_from_onnx(const std::string& onnx_path, bool enable_fp16, bool enable_int8) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_)
    );
    if (!builder) {
        std::cerr << "[TensorRTBackend] Failed to create builder" << std::endl;
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
    );
    if (!network) {
        std::cerr << "[TensorRTBackend] Failed to create network" << std::endl;
        return false;
    }

    auto parser = std::unique_ptr<nvinfer1::IOnnxParser>(
        nvinfer1::createOnnxParser(*network, logger_)
    );
    if (!parser) {
        std::cerr << "[TensorRTBackend] Failed to create ONNX parser" << std::endl;
        return false;
    }

    // Parse ONNX model
    std::ifstream file(onnx_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[TensorRTBackend] Failed to open ONNX file: " << onnx_path << std::endl;
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> model_data(size);
    file.read(model_data.data(), size);
    file.close();

    if (!parser->parse(model_data.data(), size)) {
        std::cerr << "[TensorRTBackend] Failed to parse ONNX model" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            std::cerr << "  " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    // Build config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    if (!config) {
        std::cerr << "[TensorRTBackend] Failed to create builder config" << std::endl;
        return false;
    }

    // Set workspace and precision
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
    if (enable_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (enable_int8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    // Set dynamic shapes if needed
    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();
    if (input_dims.nbDims > 0) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        nvinfer1::Dims min_dims = input_dims;
        nvinfer1::Dims opt_dims = input_dims;
        nvinfer1::Dims max_dims = input_dims;
        for (int i = 0; i < input_dims.nbDims; i++) {
            if (input_dims.d[i] == -1) {
                min_dims.d[i] = 1;
                opt_dims.d[i] = 640;
                max_dims.d[i] = 1280;
            }
        }
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
        config->addOptimizationProfile(profile);
    }

    // Build serialized engine
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config)
    );
    if (!serialized) {
        std::cerr << "[TensorRTBackend] Failed to build engine" << std::endl;
        return false;
    }

    // Create runtime and deserialize
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_),
        [](nvinfer1::IRuntime* rt) { if (rt) rt->destroy(); }
    );
    if (!runtime_) {
        std::cerr << "[TensorRTBackend] Failed to create runtime" << std::endl;
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(serialized->data(), serialized->size()),
        [](nvinfer1::ICudaEngine* eng) { if (eng) eng->destroy(); }
    );
    if (!engine_) {
        std::cerr << "[TensorRTBackend] Failed to deserialize engine" << std::endl;
        return false;
    }

    // Auto-save engine for next time
    std::string engine_path = onnx_path + ".engine";
    save_engine(engine_path);

    std::cout << "[TensorRTBackend] Built engine from ONNX: " << onnx_path << std::endl;
    return true;
}

bool TensorRTBackend::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[TensorRTBackend] Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_),
        [](nvinfer1::IRuntime* rt) { if (rt) rt->destroy(); }
    );
    if (!runtime_) {
        std::cerr << "[TensorRTBackend] Failed to create runtime" << std::endl;
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), size),
        [](nvinfer1::ICudaEngine* eng) { if (eng) eng->destroy(); }
    );
    if (!engine_) {
        std::cerr << "[TensorRTBackend] Failed to deserialize engine" << std::endl;
        return false;
    }

    std::cout << "[TensorRTBackend] Loaded engine: " << engine_path << std::endl;
    return true;
}

bool TensorRTBackend::save_engine(const std::string& engine_path) {
    auto serialized = std::shared_ptr<nvinfer1::IHostMemory>(
        engine_->serialize(),
        [](nvinfer1::IHostMemory* mem) { if (mem) mem->destroy(); }
    );
    if (!serialized) {
        std::cerr << "[TensorRTBackend] Failed to serialize engine" << std::endl;
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "[TensorRTBackend] Failed to write engine file: " << engine_path << std::endl;
        return false;
    }
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    file.close();

    std::cout << "[TensorRTBackend] Engine saved to: " << engine_path << std::endl;
    return true;
}

bool TensorRTBackend::forward(
    const std::vector<std::string>&          input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<float>&                input_data,
    const std::vector<std::string>&          output_names,
    std::vector<std::vector<int64_t>>&       output_shapes,
    std::vector<std::vector<float>>&         output_data
) {
    if (!loaded_) {
        std::cerr << "[TensorRTBackend] Model not loaded" << std::endl;
        return false;
    }

    cudaSetDevice(device_id_);

    // Set input tensor address and copy data to device
    size_t input_size = input_data.size() * sizeof(float);
    cudaMemcpy(device_buffers_[0], input_data.data(), input_size, cudaMemcpyHostToDevice);
    context_->setTensorAddress(input_names_[0].c_str(), device_buffers_[0]);

    // Set output tensor addresses
    for (size_t i = 0; i < output_names_.size(); i++) {
        context_->setTensorAddress(output_names_[i].c_str(), device_output_ptrs_[i]);
    }

    // Set dynamic input shapes if needed
    for (size_t i = 0; i < input_names.size(); i++) {
        nvinfer1::Dims dims;
        dims.nbDims = (int)input_shapes[i].size();
        for (int j = 0; j < dims.nbDims; j++) {
            dims.d[j] = (int)input_shapes[i][j];
        }
        context_->setBindingDimensions(i, dims);
    }

    // Run inference
    if (!context_->executeV2(device_buffers_)) {
        std::cerr << "[TensorRTBackend] Inference failed" << std::endl;
        return false;
    }

    // Copy outputs back to host
    output_data.resize(output_names_.size());
    output_shapes.resize(output_names_.size());

    for (size_t i = 0; i < output_names_.size(); i++) {
        // Find matching output
        int idx = -1;
        for (size_t j = 0; j < output_names_.size(); j++) {
            if (output_names_[j] == output_names[i]) {
                idx = (int)j;
                break;
            }
        }
        if (idx < 0) {
            std::cerr << "[TensorRTBackend] Output not found: " << output_names[i] << std::endl;
            return false;
        }

        output_shapes[i] = output_shapes_[idx];
        size_t total_elements = output_sizes_[idx];
        output_data[i].resize(total_elements);
        cudaMemcpy(output_data[i].data(), device_output_ptrs_[idx],
                   total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    }

    return true;
}

std::vector<std::string> TensorRTBackend::get_input_names() const {
    return input_names_;
}

std::vector<std::vector<int64_t>> TensorRTBackend::get_input_shapes() const {
    return input_shapes_;
}

std::vector<std::string> TensorRTBackend::get_output_names() const {
    return output_names_;
}

std::vector<std::vector<int64_t>> TensorRTBackend::get_output_shapes() const {
    return output_shapes_;
}

} // namespace yolo_onnx