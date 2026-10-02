#include "core/openvino_backend.hpp"

#include <openvino/openvino.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

namespace yolo_onnx {

// ============================================================
// OpenvinoBackend
// ============================================================
// 直接加载 .onnx（OpenVINO 内部完成 IR 转换与图优化），
// 通过 custom_config 的 device=CPU/GPU/AUTO/... 选择目标设备。
// Core 实例创建开销较大（枚举插件），全进程共享一个。
// ============================================================

namespace {
ov::Core& shared_core() {
    static ov::Core core;
    static std::once_flag once;
    std::call_once(once, [] { /* 惰性初始化，首次调用时构造 */ });
    return core;
}

/// 把 OV 的 element type 转成 OpenVINO 的数值（仅支持 float 系列）
bool to_float_data(const ov::Tensor& t, cv::Mat& out) {
    switch (t.get_element_type()) {
        case ov::element::f32: {
            out = cv::Mat(1, (int)t.get_size(), CV_32F);
            std::memcpy(out.ptr<float>(), t.data(), t.get_size() * sizeof(float));
            return true;
        }
        default:
            return false;
    }
}
} // namespace

struct OpenvinoBackend::Impl {
    std::shared_ptr<ov::Model>              model;
    ov::CompiledModel                       compiled;
    ov::InferRequest                        request;

    std::vector<std::string>                input_names;
    std::vector<std::vector<int64_t>>       input_shapes;
    std::vector<std::string>                output_names;
    std::vector<std::vector<int64_t>>       output_shapes;
};

OpenvinoBackend::OpenvinoBackend()  : impl_(std::make_unique<Impl>()) {}
OpenvinoBackend::~OpenvinoBackend() = default;

std::string OpenvinoBackend::parse_device(const std::string& custom_config) const {
    // 支持: device=CPU / device=GPU / device=GPU.1；缺省 CPU
    std::istringstream stream(custom_config);
    std::string token;
    while (std::getline(stream, token, ';')) {
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = token.substr(0, eq);
        std::string val = token.substr(eq + 1);
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key); trim(val);
        if (key == "device" || key == "--device") {
            // 纯数字当作设备编号 → GPU.<id>；字符串当作设备名
            if (!val.empty() && std::all_of(val.begin(), val.end(), ::isdigit)) {
                return "GPU." + val;
            }
            return val;
        }
    }
    return "CPU";
}

bool OpenvinoBackend::load(const Config& config) {
    try {
        auto& core = shared_core();
        const std::string device = parse_device(config.custom_config);

        impl_->model = core.read_model(config.model_path);

        ov::AnyMap props;
        if (config.num_threads > 0 && device.find("CPU") != std::string::npos) {
            props["INFERENCE_NUM_THREADS"] = config.num_threads;
        }
        impl_->compiled = core.compile_model(impl_->model, device, props);
        impl_->request  = impl_->compiled.create_infer_request();

        auto collect = [](const ov::Output<const ov::Node>& port,
                          std::string& name,
                          std::vector<int64_t>& shape) {
            name = port.get_any_name();
            shape.clear();
            for (auto d : port.get_partial_shape()) {
                shape.push_back(d.is_dynamic() ? -1 : d.get_length());
            }
        };

        impl_->input_names.clear();  impl_->input_shapes.clear();
        impl_->output_names.clear(); impl_->output_shapes.clear();
        for (const auto& in : impl_->compiled.inputs()) {
            std::string n; std::vector<int64_t> s;
            collect(in, n, s);
            impl_->input_names.push_back(n);
            impl_->input_shapes.push_back(s);
        }
        for (const auto& out : impl_->compiled.outputs()) {
            std::string n; std::vector<int64_t> s;
            collect(out, n, s);
            impl_->output_names.push_back(n);
            impl_->output_shapes.push_back(s);
        }

        name_ = "openvino/" + device;
        loaded_ = true;
        std::cout << "[OpenvinoBackend] Using device: " << device
                  << " (model: " << config.model_path << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OpenvinoBackend] Failed to load model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

bool OpenvinoBackend::forward(
    const std::vector<std::string>&          input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<cv::Mat>&              input_data,
    const std::vector<std::string>&          output_names,
    TensorSet&                               outputs
) {
    if (!loaded_) {
        std::cerr << "[OpenvinoBackend] Model not loaded" << std::endl;
        return false;
    }
    try {
        // 绑定输入（按名字；cv::Mat 是连续 NCHW float32 缓冲）
        for (size_t i = 0; i < input_names.size(); i++) {
            const cv::Mat& m = input_data[i];
            size_t total = static_cast<size_t>(m.total() * m.channels());

            ov::Tensor t(ov::element::f32,
                         ov::Shape(input_shapes[i].begin(), input_shapes[i].end()));
            std::memcpy(t.data(), m.ptr<float>(), total * sizeof(float));
            impl_->request.set_tensor(input_names[i], t);
        }

        impl_->request.infer();

        // 收集输出
        outputs.datas.clear();
        outputs.shapes.clear();
        for (size_t i = 0; i < output_names.size(); i++) {
            ov::Tensor t = impl_->request.get_tensor(output_names[i]);

            ov::Shape s = t.get_shape();
            std::vector<int64_t> shape(s.begin(), s.end());

            cv::Mat mat;
            if (!to_float_data(t, mat)) {
                std::cerr << "[OpenvinoBackend] Unsupported output dtype for '"
                          << output_names[i] << "'" << std::endl;
                return false;
            }
            outputs.datas.push_back(mat);
            outputs.shapes.push_back(shape);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OpenvinoBackend] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> OpenvinoBackend::get_input_names() const {
    return impl_->input_names;
}
std::vector<std::vector<int64_t>> OpenvinoBackend::get_input_shapes() const {
    return impl_->input_shapes;
}
std::vector<std::string> OpenvinoBackend::get_output_names() const {
    return impl_->output_names;
}
std::vector<std::vector<int64_t>> OpenvinoBackend::get_output_shapes() const {
    return impl_->output_shapes;
}

} // namespace yolo_onnx
