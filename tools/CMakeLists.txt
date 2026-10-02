# tools/ — comparison / debug utilities (not part of the installed library)
add_executable(dump_json dump_json.cpp)
target_link_libraries(dump_json yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})