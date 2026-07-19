# Build examples
add_executable(detect_image detect_image.cpp)
target_link_libraries(detect_image yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
install(TARGETS detect_image DESTINATION bin)