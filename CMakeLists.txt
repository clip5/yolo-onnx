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
)

# ---- Sources ----
set(YOLO_ONNX_SOURCES
    src/core/onnxruntime_backend.cpp
    src/process/preprocess/preprocess.cpp
    src/process/postprocess/postprocess.cpp
    src/model.cpp
    src/models/detect/v5.cpp
    src/models/detect/yolox.cpp
    src/models/detect/v8.cpp
    src/models/detect/ppyoloe.cpp
    src/models/segment/v8.cpp
    src/models/pose/v8.cpp
    src/models/obb/v8.cpp
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