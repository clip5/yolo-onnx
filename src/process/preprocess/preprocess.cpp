#include "process/preprocess/preprocess.hpp"
#include <opencv2/imgproc.hpp>

namespace yolo_onnx {

PreProcessParams PreProcessParams::for_model(ModelType type, int width, int height) {
    PreProcessParams p;
    p.target_width  = width;
    p.target_height = height;
    // PPYOLOE 输入保持 BGR 顺序，其余 YOLO 系为 RGB
    p.swap_rb = (type != ModelType::PPYOLOE);
    return p;
}

PreProcess::PreProcess(const PreProcessParams& params) : params_(params) {}

PreProcessResult PreProcess::run(const cv::Mat& image) const {
    PreProcessResult result;

    const int target_w = params_.target_width;
    const int target_h = params_.target_height;
    const int img_w = image.cols;
    const int img_h = image.rows;

    // 1. Letterbox resize (maintain aspect ratio with padding)
    float scale = std::min((float)target_w / img_w, (float)target_h / img_h);
    int new_w = (int)(img_w * scale);
    int new_h = (int)(img_h * scale);
    int pad_left = (target_w - new_w) / 2;
    int pad_top  = (target_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_left, pad_top, new_w, new_h)));

    result.letterbox = {
        scale, pad_left, pad_top,
        img_w, img_h, target_w, target_h
    };

    // 2. Convert to NCHW float blob (optional BGR→RGB, scale, per-channel mean/std)
    result.blob.resize(3 * (size_t)target_h * target_w);
    for (int c = 0; c < 3; c++) {
        // blob 通道 0 固定为 R 语义；swap_rb=false 时按 BGR 语义取源通道
        int src_c = params_.swap_rb ? (2 - c) : c;
        for (int h = 0; h < target_h; h++) {
            for (int w = 0; w < target_w; w++) {
                float v = canvas.at<cv::Vec3b>(h, w)[src_c] * params_.scale_factor;
                v = (v - params_.mean[c]) / params_.std[c];
                result.blob[(size_t)c * target_h * target_w + h * target_w + w] = v;
            }
        }
    }

    return result;
}

} // namespace yolo_onnx
