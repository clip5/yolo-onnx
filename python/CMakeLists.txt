# Python bindings (pybind11)
# Enable with: cmake .. -DWITH_PYTHON=ON
# Requires: find_package(pybind11 REQUIRED)

find_package(pybind11 REQUIRED)

# Source files for the Python module
set(PYTHON_BINDING_SOURCES
    ../src/core/onnxruntime_backend.cpp
    ../src/process/preprocess/preprocess.cpp
    ../src/process/postprocess/postprocess.cpp
    ../src/model.cpp
    ../src/models/detect/v5.cpp
    ../src/models/detect/yolox.cpp
    ../src/models/detect/v8.cpp
    ../src/models/detect/ppyoloe.cpp
    ../src/models/segment/v8.cpp
    ../src/models/pose/v8.cpp
    ../src/models/obb/v8.cpp
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