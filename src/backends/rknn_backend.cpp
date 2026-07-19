#include "yolo_onnx/backends/rknn_backend.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>

#include "rknn_api.h"

namespace yolo_onnx {

RKNNBackend::RKNNBackend()
    : ctx_(nullptr) {
}

RKNNBackend::~RKNNBackend() {
    if (ctx_) {
        rknn_destroy(ctx_);
        ctx_ = nullptr;
    }
}

bool RKNNBackend::load(const Config& config) {
    // Read RKNN model file
    std::ifstream file(config.model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[RKNNBackend] Failed to open model file: " << config.model_path << std::endl;
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(size);
    file.read(reinterpret_cast<char*>(model_data.data()), size);
    file.close();

    // Initialize RKNN
    ctx_ = new rknn_context();
    if (!ctx_) {
        std::cerr << "[RKNNBackend] Failed to allocate context" << std::endl;
        return false;
    }
    *ctx_ = 0;

    int ret = rknn_init(ctx_, model_data.data(), size, 0, nullptr);
    if (ret < 0) {
        std::cerr << "[RKNNBackend] rknn_init failed: " << ret << std::endl;
        delete ctx_;
        ctx_ = nullptr;
        return false;
    }

    // Query input/output info
    // Get input/output count
    ret = rknn_query(*ctx_, RKNN_QUERY_IN_OUT_NUM, nullptr, 0);
    if (ret < 0) {
        std::cerr << "[RKNNBackend] Failed to query I/O count" << std::endl;
        rknn_destroy(ctx_);
        delete ctx_;
        ctx_ = nullptr;
        return false;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(*ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cerr << "[RKNNBackend] Failed to query I/O count" << std::endl;
        rknn_destroy(ctx_);
        delete ctx_;
        ctx_ = nullptr;
        return false;
    }

    num_inputs_ = io_num.n_input;
    num_outputs_ = io_num.n_output;

    // Query input tensors
    std::vector<rknn_tensor_attr> input_attrs(num_inputs_);
    for (int i = 0; i < num_inputs_; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(*ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cerr << "[RKNNBackend] Failed to query input attr " << i << std::endl;
            rknn_destroy(ctx_);
            delete ctx_;
            ctx_ = nullptr;
            return false;
        }
    }

    // Query output tensors
    std::vector<rknn_tensor_attr> output_attrs(num_outputs_);
    for (int i = 0; i < num_outputs_; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(*ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cerr << "[RKNNBackend] Failed to query output attr " << i << std::endl;
            rknn_destroy(ctx_);
            delete ctx_;
            ctx_ = nullptr;
            return false;
        }
    }

    // Populate input shapes
    input_names_.resize(num_inputs_);
    input_shapes_.resize(num_inputs_);
    for (int i = 0; i < num_inputs_; i++) {
        input_names_[i] = input_attrs[i].name ? input_attrs[i].name : ("input_" + std::to_string(i));
        std::vector<int64_t> shape;
        for (uint32_t d = 0; d < input_attrs[i].n_dims; d++) {
            shape.push_back(input_attrs[i].dims[d]);
        }
        input_shapes_[i] = shape;
    }

    // Populate output shapes
    output_names_.resize(num_outputs_);
    output_shapes_.resize(num_outputs_);
    for (int i = 0; i < num_outputs_; i++) {
        output_names_[i] = output_attrs[i].name ? output_attrs[i].name : ("output_" + std::to_string(i));
        std::vector<int64_t> shape;
        for (uint32_t d = 0; d < output_attrs[i].n_dims; d++) {
            shape.push_back(output_attrs[i].dims[d]);
        }
        output_shapes_[i] = shape;
    }

    loaded_ = true;

    std::cout << "[RKNNBackend] Model loaded: " << config.model_path
              << " (inputs=" << num_inputs_ << ", outputs=" << num_outputs_ << ")"
              << std::endl;
    return true;
}

bool RKNNBackend::forward(
    const std::vector<std::string>&          input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<float>&                input_data,
    const std::vector<std::string>&          output_names,
    std::vector<std::vector<int64_t>>&       output_shapes,
    std::vector<std::vector<float>>&         output_data
) {
    if (!loaded_ || !ctx_) {
        std::cerr << "[RKNNBackend] Model not loaded" << std::endl;
        return false;
    }

    // Prepare input tensors
    std::vector<rknn_input> inputs(num_inputs_);
    std::memset(inputs.data(), 0, sizeof(rknn_input) * num_inputs_);

    size_t data_offset = 0;
    for (int i = 0; i < num_inputs_; i++) {
        // Calculate total elements for this input
        size_t total_elements = 1;
        for (auto& d : input_shapes[i]) {
            total_elements *= d;
        }

        inputs[i].index = i;
        inputs[i].type = RKNN_TENSOR_FLOAT32;
        inputs[i].size = total_elements * sizeof(float);
        inputs[i].fmt = RKNN_TENSOR_NCHW;
        inputs[i].buf = const_cast<float*>(input_data.data() + data_offset);
        inputs[i].pass_through = 0;  // Apply quantization if needed

        data_offset += total_elements;
    }

    int ret = rknn_inputs_set(*ctx_, num_inputs_, inputs.data());
    if (ret < 0) {
        std::cerr << "[RKNNBackend] rknn_inputs_set failed: " << ret << std::endl;
        return false;
    }

    // Run inference
    ret = rknn_run(*ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[RKNNBackend] rknn_run failed: " << ret << std::endl;
        return false;
    }

    // Get output tensors
    std::vector<rknn_output> outputs(num_outputs_);
    std::memset(outputs.data(), 0, sizeof(rknn_output) * num_outputs_);
    for (int i = 0; i < num_outputs_; i++) {
        outputs[i].want_float = 1;  // Dequantize to float
    }

    ret = rknn_outputs_get(*ctx_, num_outputs_, outputs.data(), nullptr);
    if (ret < 0) {
        std::cerr << "[RKNNBackend] rknn_outputs_get failed: " << ret << std::endl;
        return false;
    }

    // Copy output data
    output_data.resize(num_outputs_);
    output_shapes.resize(num_outputs_);

    for (int i = 0; i < num_outputs_; i++) {
        output_shapes[i] = output_shapes_[i];

        size_t total_elements = outputs[i].size / sizeof(float);
        output_data[i].resize(total_elements);
        std::memcpy(output_data[i].data(), outputs[i].buf, outputs[i].size);
    }

    // Release outputs
    rknn_outputs_release(*ctx_, num_outputs_, outputs.data());

    return true;
}

std::vector<std::string> RKNNBackend::get_input_names() const {
    return input_names_;
}

std::vector<std::vector<int64_t>> RKNNBackend::get_input_shapes() const {
    return input_shapes_;
}

std::vector<std::string> RKNNBackend::get_output_names() const {
    return output_names_;
}

std::vector<std::vector<int64_t>> RKNNBackend::get_output_shapes() const {
    return output_shapes_;
}

} // namespace yolo_onnx