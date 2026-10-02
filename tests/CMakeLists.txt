# Build tests
add_executable(test_yolo test_yolo.cpp)

# tests 直接包含 src/ 下的 postprocess_core.hpp —— 既测试后处理基元，
# 也验证该头文件可脱离 yolo_onnx.hpp 独立使用（可整份拷到其他项目）。
target_include_directories(test_yolo PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_yolo yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
install(TARGETS test_yolo DESTINATION bin)

# Register test
add_test(NAME test_yolo COMMAND test_yolo)