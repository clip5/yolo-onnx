# Build tests
add_executable(test_yolo test_yolo.cpp)
target_link_libraries(test_yolo yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
install(TARGETS test_yolo DESTINATION bin)

# Register test
add_test(NAME test_yolo COMMAND test_yolo)