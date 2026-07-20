# Python bindings (pybind11)
# Enable with: cmake .. -DWITH_PYTHON=ON
# Requires: find_package(pybind11 REQUIRED)

find_package(pybind11 REQUIRED)

# Source files for the Python module
set(PYTHON_BINDING_SOURCES
    ../src/backends/onnxruntime_backend.cpp
    ../src/model.cpp
    ../src/models/model_v5.cpp
    ../src/models/model_yolox.cpp
    ../src/models/model_v8.cpp
    ../src/models/model_ppyoloe.cpp
    ../src/models/model_v8_segment.cpp
    ../src/models/model_v8_pose.cpp
    ../src/models/model_v8_obb.cpp
    python_bindings.cpp
)

pybind11_add_module(yolo_onnx_py ${PYTHON_BINDING_SOURCES})

target_include_directories(yolo_onnx_py
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${PYTHON_BINDING_INCLUDE_DIRS}
)

target_link_libraries(yolo_onnx_py
    PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBRARIES}
        ${PYTHON_BINDING_LINK_LIBS}
        pthread
)

# Set output directory so the module is placed alongside the shared library
set_target_properties(yolo_onnx_py
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)