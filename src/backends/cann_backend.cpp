#include "yolo_onnx/backends/cann_backend.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cstdint>

// AscendCL headers
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

namespace yolo_onnx {

CANNBackend::CANNBackend()
    : context_(nullptr)
    , stream_(nullptr)
    , model_id_(0)
    , input_buffer_(nullptr)
    , output_buffer_(nullptr) {
}

CANNBackend::~CANNBackend() {
    destroy_acl();
}

bool CANNBackend::init_acl() {
    // Initialize AscendCL
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] aclInit failed: " << ret << std::endl;
        return false;
    }

    // Set device
    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] aclrtSetDevice failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }

    // Create context
    ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] aclrtCreateContext failed: " << ret << std::endl;
        aclrtResetDevice(device_id_);
        aclFinalize();
        return false;
    }

    // Create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] aclrtCreateStream failed: " << ret << std::endl;
        aclrtDestroyContext(context_);
        context_ = nullptr;
        aclrtResetDevice(device_id_);
        aclFinalize();
        return false;
    }

    return true;
}

void CANNBackend::destroy_acl() {
    if (input_buffer_) {
        aclrtFree(input_buffer_);
        input_buffer_ = nullptr;
    }
    if (output_buffer_) {
        aclrtFree(output_buffer_);
        output_buffer_ = nullptr;
    }
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool CANNBackend::load(const Config& config) {
    device_id_ = config.device_id;

    // Initialize AscendCL
    if (!init_acl()) {
        return false;
    }

    // Load OM model
    aclError ret = aclmdlLoadFromFile(config.model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] Failed to load model: " << config.model_path
                  << " (error: " << ret << ")" << std::endl;
        destroy_acl();
        return false;
    }

    // Get model description
    aclmdlDesc* model_desc = aclmdlCreateDesc();
    ret = aclmdlGetDesc(model_desc, model_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] Failed to get model description" << std::endl;
        aclmdlDestroyDesc(model_desc);
        destroy_acl();
        return false;
    }

    // Get input info
    size_t num_inputs = aclmdlGetNumInputs(model_desc);
    input_names_.resize(num_inputs);
    input_shapes_.resize(num_inputs);
    input_size_ = 0;

    for (size_t i = 0; i < num_inputs; i++) {
        auto dims = aclmdlGetInputDims(model_desc, i);
        input_names_[i] = "input_" + std::to_string(i);

        std::vector<int64_t> shape(dims->dims, dims->dims + dims->dimCount);
        input_shapes_[i] = shape;

        // Calculate size
        size_t dim_size = aclmdlGetInputSizeByIndex(model_desc, i);
        input_size_ += dim_size;
    }

    // Get output info
    size_t num_outputs = aclmdlGetNumOutputs(model_desc);
    output_names_.resize(num_outputs);
    output_shapes_.resize(num_outputs);
    output_size_ = 0;

    for (size_t i = 0; i < num_outputs; i++) {
        auto dims = aclmdlGetOutputDims(model_desc, i);
        output_names_[i] = "output_" + std::to_string(i);

        std::vector<int64_t> shape(dims->dims, dims->dims + dims->dimCount);
        output_shapes_[i] = shape;

        size_t dim_size = aclmdlGetOutputSizeByIndex(model_desc, i);
        output_size_ += dim_size;
    }

    aclmdlDestroyDesc(model_desc);

    // Allocate device memory
    if (input_size_ > 0) {
        ret = aclrtMalloc(&input_buffer_, input_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[CANNBackend] Failed to allocate input buffer" << std::endl;
            destroy_acl();
            return false;
        }
    }

    if (output_size_ > 0) {
        ret = aclrtMalloc(&output_buffer_, output_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[CANNBackend] Failed to allocate output buffer" << std::endl;
            destroy_acl();
            return false;
        }
    }

    loaded_ = true;

    std::cout << "[CANNBackend] Model loaded: " << config.model_path
              << " (inputs=" << num_inputs << ", outputs=" << num_outputs << ")"
              << std::endl;
    return true;
}

bool CANNBackend::forward(
    const std::vector<std::string>&          input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<float>&                input_data,
    const std::vector<std::string>&          output_names,
    std::vector<std::vector<int64_t>>&       output_shapes,
    std::vector<std::vector<float>>&         output_data
) {
    if (!loaded_) {
        std::cerr << "[CANNBackend] Model not loaded" << std::endl;
        return false;
    }

    // Copy input data to device
    aclError ret = aclrtMemcpy(
        input_buffer_, input_size_,
        input_data.data(), input_data.size() * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] Failed to copy input data to device" << std::endl;
        return false;
    }

    // Prepare input/output buffers for aclmdlExecute
    std::vector<aclDataBuffer*> input_buffers;
    aclDataBuffer* input_data_buf = aclCreateDataBuffer(input_buffer_, input_size_);
    input_buffers.push_back(input_data_buf);

    // Create aclmdlDataset for inputs
    aclmdlDataset* input_dataset = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(input_dataset, input_data_buf);

    // Prepare output buffers
    aclmdlDataset* output_dataset = aclmdlCreateDataset();
    aclDataBuffer* output_data_buf = aclCreateDataBuffer(output_buffer_, output_size_);
    aclmdlAddDatasetBuffer(output_dataset, output_data_buf);

    // Execute model
    ret = aclmdlExecute(model_id_, input_dataset, output_dataset);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[CANNBackend] Inference failed: " << ret << std::endl;
        aclmdlDestroyDataset(input_dataset);
        aclmdlDestroyDataset(output_dataset);
        aclDestroyDataBuffer(input_data_buf);
        aclDestroyDataBuffer(output_data_buf);
        return false;
    }

    // Copy output data back to host
    output_data.resize(output_names_.size());
    output_shapes.resize(output_names_.size());

    size_t offset = 0;
    for (size_t i = 0; i < output_names_.size(); i++) {
        output_shapes[i] = output_shapes_[i];

        // Calculate number of elements from shape
        size_t total_elements = 1;
        for (auto& d : output_shapes_[i]) {
            total_elements *= d;
        }

        // Copy from device buffer (each output is float32)
        output_data[i].resize(total_elements);
        size_t copy_size = total_elements * sizeof(float);

        // Clamp to available device buffer
        if (offset + copy_size > output_size_) {
            copy_size = output_size_ - offset;
        }

        ret = aclrtMemcpy(
            output_data[i].data(), copy_size,
            (uint8_t*)output_buffer_ + offset, copy_size,
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[CANNBackend] Failed to copy output data from device" << std::endl;
            aclmdlDestroyDataset(input_dataset);
            aclmdlDestroyDataset(output_dataset);
            aclDestroyDataBuffer(input_data_buf);
            aclDestroyDataBuffer(output_data_buf);
            return false;
        }

        offset += copy_size;
    }

    // Cleanup
    aclmdlDestroyDataset(input_dataset);
    aclmdlDestroyDataset(output_dataset);
    aclDestroyDataBuffer(input_data_buf);
    aclDestroyDataBuffer(output_data_buf);

    return true;
}

std::vector<std::string> CANNBackend::get_input_names() const {
    return input_names_;
}

std::vector<std::vector<int64_t>> CANNBackend::get_input_shapes() const {
    return input_shapes_;
}

std::vector<std::string> CANNBackend::get_output_names() const {
    return output_names_;
}

std::vector<std::vector<int64_t>> CANNBackend::get_output_shapes() const {
    return output_shapes_;
}

} // namespace yolo_onnx