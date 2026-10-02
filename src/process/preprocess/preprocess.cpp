#include "process/preprocess/preprocess.hpp"
#include <opencv2/imgproc.hpp>

namespace yolo_onnx {

PreProcessParams PreProcessParams::for_model(ModelType type, int width, int height) {
    PreProcessParams p;
    p.target_width  = width;
    p.target_height = height;
    p.swap_rb = true;

    switch (type) {
        // PPYOLOE 输入保持 BGR 顺序
        case ModelType::PPYOLOE:
            p.swap_rb = false;
            break;
        // YOLOX 使用 ImageNet mean/std 归一化
        case ModelType::YOLOX: {
            static const float mean[3] = {0.485f, 0.456f, 0.406f};
            static const float std_[3] = {0.229f, 0.224f, 0.225f};
            for (int c = 0; c < 3; c++) { p.mean[c] = mean[c]; p.std[c] = std_[c]; }
            break;
        }
        default:
            break;
    }
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

    // 复用画布：尺寸不变时不重新分配，只重写像素内容
    if (canvas_.empty() || canvas_.size() != cv::Size(target_w, target_h)) {
        canvas_.create(target_h, target_w, CV_8UC3);
    }
    canvas_.setTo(cv::Scalar(114, 114, 114));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));
    resized.copyTo(canvas_(cv::Rect(pad_left, pad_top, new_w, new_h)));

    result.letterbox = {
        scale, pad_left, pad_top,
        img_w, img_h, target_w, target_h
    };

    // 2. 打包为 NCHW float32 (可选 BGR→RGB, scale, per-channel mean/std)
    //    cv::split / convertTo 均属 OpenCV core 模块（不依赖 dnn），走 SIMD 优化路径。
    result.blob = make_nchw_blob(1, 3, target_h, target_w);

    channels_.resize(3);
    cv::split(canvas_, channels_);   // channels_[0]=B, [1]=G, [2]=R

    for (int c = 0; c < 3; c++) {
        // blob 通道 0 固定为 R 语义；swap_rb=false 时按 BGR 语义取源通道
        const int src_c = params_.swap_rb ? (2 - c) : c;
        // 单通道视图：直接写入最终 blob，无中间 Mat、无 memcpy
        cv::Mat dst(target_h, target_w, CV_32F, result.blob.ptr<float>(0, c, 0));
        // (v * scale_factor - mean) / std  折成  convertTo 的  v * alpha + beta
        const float alpha = params_.scale_factor / params_.std[c];
        const float beta  = -params_.mean[c] / params_.std[c];
        channels_[src_c].convertTo(dst, CV_32F, alpha, beta);
    }

    return result;
}

} // namespace yolo_onnx