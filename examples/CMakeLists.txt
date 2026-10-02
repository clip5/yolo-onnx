# Build examples
add_executable(detect_image detect_image.cpp)
target_link_libraries(detect_image yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})

add_executable(segment_image segment_image.cpp)
target_link_libraries(segment_image yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})

add_executable(pose_image pose_image.cpp)
target_link_libraries(pose_image yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})

add_executable(obb_image obb_image.cpp)
target_link_libraries(obb_image yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})

add_executable(sem_image sem_image.cpp)
target_link_libraries(sem_image yolo_onnx ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})

install(TARGETS detect_image segment_image pose_image obb_image sem_image DESTINATION bin)