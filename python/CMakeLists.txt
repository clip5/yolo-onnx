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

# Optional TensorRT backend
if(WITH_TENSORRT AND TensorRT_FOUND)
    list(APPEND PYTHON_BINDING_SOURCES ../src/backends/tensorrt_backend.cpp)
    list(APPEND PYTHON_BINDING_INCLUDE_DIRS ${TensorRT_INCLUDE_DIRS})
    list(APPEND PYTHON_BINDING_LINK_LIBS ${TensorRT_LIBRARIES})
    add_definitions(-DWITH_TENSORRT_BACKEND)
endif()

# Optional CANN backend
if(WITH_CANN AND CANN_FOUND)
    list(APPEND PYTHON_BINDING_SOURCES ../src/backends/cann_backend.cpp)
    list(APPEND PYTHON_BINDING_INCLUDE_DIRS ${CANN_INCLUDE_DIRS})
    list(APPEND PYTHON_BINDING_LINK_LIBS ${CANN_LIBRARIES})
    add_definitions(-DWITH_CANN_BACKEND)
endif()

# Optional RKNN backend
if(WITH_RKNN AND RKNN_FOUND)
    list(APPEND PYTHON_BINDING_SOURCES ../src/backends/rknn_backend.cpp)
    list(APPEND PYTHON_BINDING_INCLUDE_DIRS ${RKNN_INCLUDE_DIRS})
    list(APPEND PYTHON_BINDING_LINK_LIBS ${RKNN_LIBRARIES})
    add_definitions(-DWITH_RKNN_BACKEND)
endif()

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