#include "process/postprocess/decoder/decoder.hpp"

#include <iostream>

namespace yolo_onnx {

// ============================================================
// Decoder 工厂：按模型类型选择输出格式解码器
// ============================================================
// 新增模型格式时只需在这里加一个分支，无需改动 Model 或 PostProcess。
std::shared_ptr<Decoder> create_decoder(ModelType type) {
    switch (type) {
        case ModelType::YOLOv5:   return std::make_shared<V5Decoder>();
        case ModelType::YOLOX:    return std::make_shared<YOLOXDecoder>();
        case ModelType::PPYOLOE:  return std::make_shared<PPYOLOEDecoder>();
        // v8 / v11 / v26 输出格式一致，共用同一解码器
        case ModelType::YOLOv8:
        case ModelType::YOLOv11:
        case ModelType::YOLO26:   return std::make_shared<V8Decoder>();
        default:
            std::cerr << "[create_decoder] No decoder for model type: "
                      << model_type_name(type) << std::endl;
            return nullptr;
    }
}

} // namespace yolo_onnx