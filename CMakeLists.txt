cmake_minimum_required(VERSION 3.16)
project(yolo-onnx VERSION 0.1.0 LANGUAGES CXX)

# ============================================================
# Project: yolo-onnx — YOLO inference framework based on ONNX Runtime
# Supports: YOLOv5, YOLOX, YOLOv8, YOLOv11, YOLO26, PPYOLOE
# Backends: onnxruntime (current), tensorrt / npu / cann (future)
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Options ----
option(WITH_EXAMPLES "Build examples" ON)
option(WITH_TESTS    "Build tests"    OFF)
option(WITH_PYTHON   "Build python bindings" OFF)

# Backend options
option(WITH_TENSORRT "Build TensorRT backend (NVIDIA GPU)" OFF)
option(WITH_CANN     "Build CANN backend (Huawei Ascend NPU)" OFF)
option(WITH_RKNN     "Build RKNN backend (Rockchip NPU)" OFF)

# ---- Dependencies ----
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# ONNX Runtime (custom find module)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(ONNXRuntime REQUIRED)

# ---- Include directories ----
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${YOLO_ONNX_INCLUDE_DIRS}
)

# ---- Backend sources ----
# Always include onnxruntime backend
set(YOLO_ONNX_BACKEND_SOURCES
    src/backends/onnxruntime_backend.cpp
)

# Optional TensorRT backend
if(WITH_TENSORRT)
    find_package(TensorRT)
    if(TensorRT_FOUND)
        list(APPEND YOLO_ONNX_BACKEND_SOURCES src/backends/tensorrt_backend.cpp)
        list(APPEND YOLO_ONNX_INCLUDE_DIRS ${TensorRT_INCLUDE_DIRS})
        list(APPEND YOLO_ONNX_LINK_LIBS ${TensorRT_LIBRARIES})
        add_definitions(-DWITH_TENSORRT_BACKEND)
        message(STATUS "TensorRT backend: ENABLED")
    else()
        message(WARNING "WITH_TENSORRT=ON but TensorRT not found - disabling")
    endif()
endif()

# Optional CANN backend
if(WITH_CANN)
    find_package(CANN)
    if(CANN_FOUND)
        list(APPEND YOLO_ONNX_BACKEND_SOURCES src/backends/cann_backend.cpp)
        list(APPEND YOLO_ONNX_INCLUDE_DIRS ${CANN_INCLUDE_DIRS})
        list(APPEND YOLO_ONNX_LINK_LIBS ${CANN_LIBRARIES})
        add_definitions(-DWITH_CANN_BACKEND)
        message(STATUS "CANN backend: ENABLED")
    else()
        message(WARNING "WITH_CANN=ON but CANN not found - disabling")
    endif()
endif()

# Optional RKNN backend
if(WITH_RKNN)
    find_package(RKNN)
    if(RKNN_FOUND)
        list(APPEND YOLO_ONNX_BACKEND_SOURCES src/backends/rknn_backend.cpp)
        list(APPEND YOLO_ONNX_INCLUDE_DIRS ${RKNN_INCLUDE_DIRS})
        list(APPEND YOLO_ONNX_LINK_LIBS ${RKNN_LIBRARIES})
        add_definitions(-DWITH_RKNN_BACKEND)
        message(STATUS "RKNN backend: ENABLED")
    else()
        message(WARNING "WITH_RKNN=ON but RKNN not found - disabling")
    endif()
endif()

# ---- Sources ----
set(YOLO_ONNX_SOURCES
    ${YOLO_ONNX_BACKEND_SOURCES}
    src/model.cpp
    src/models/model_v5.cpp
    src/models/model_yolox.cpp
    src/models/model_v8.cpp
    src/models/model_ppyoloe.cpp
    src/models/model_v8_segment.cpp
    src/models/model_v8_pose.cpp
    src/models/model_v8_obb.cpp
)

# ---- Library ----
add_library(yolo_onnx SHARED ${YOLO_ONNX_SOURCES})

target_include_directories(yolo_onnx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(yolo_onnx
    PUBLIC
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBRARIES}
        ${YOLO_ONNX_LINK_LIBS}
    PRIVATE
        pthread
)

# ---- Install ----
install(TARGETS yolo_onnx
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/yolo_onnx
    DESTINATION include
)

# ---- Subdirectories ----
if(WITH_EXAMPLES)
    add_subdirectory(examples)
endif()

if(WITH_TESTS)
    add_subdirectory(tests)
endif()

if(WITH_PYTHON)
    add_subdirectory(python)
endif()