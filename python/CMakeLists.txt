# Python bindings (pybind11) — placeholder
# Enable with: cmake .. -DWITH_PYTHON=ON
# Requires: find_package(pybind11 REQUIRED)

# pybind11_add_module(yolo_onnx_py
#     ../src/backends/onnxruntime_backend.cpp
#     ../src/model.cpp
#     ../src/models/model_v5.cpp
#     ../src/models/model_yolox.cpp
#     ../src/models/model_v8.cpp
#     ../src/models/model_ppyoloe.cpp
#     python_bindings.cpp
# )
# target_include_directories(yolo_onnx_py PRIVATE ../include)
# target_link_libraries(yolo_onnx_py PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})