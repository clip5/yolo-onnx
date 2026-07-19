# FindTensorRT.cmake
# Locate NVIDIA TensorRT library and headers.
#
# Sets:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_LIBRARY_DIRS
#
# Environment variables:
#   TENSORRT_DIR  (hint path)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS
        /usr/include
        /usr/local/include
        /usr/include/x86_64-linux-gnu
        /opt/TensorRT/include
        /opt/tensorrt/include
        $ENV{TENSORRT_DIR}/include
        $ENV{TENSORRT_ROOT}/include
        $ENV{TRT_ROOT}/include
)

find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer libnvinfer
    PATHS
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/lib64
        /opt/TensorRT/lib
        /opt/tensorrt/lib
        $ENV{TENSORRT_DIR}/lib
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TRT_ROOT}/lib
)

find_library(TensorRT_NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin libnvinfer_plugin
    PATHS
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/lib64
        /opt/TensorRT/lib
        /opt/tensorrt/lib
        $ENV{TENSORRT_DIR}/lib
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TRT_ROOT}/lib
)

find_library(TensorRT_CUDART_LIBRARY
    NAMES cudart libcudart
    PATHS
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
        $ENV{CUDA_DIR}/lib64
        $ENV{CUDA_HOME}/lib64
        $ENV{CUDA_PATH}/lib64
)

find_library(TensorRT_CUBLAS_LIBRARY
    NAMES cublas libcublas
    PATHS
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
        $ENV{CUDA_DIR}/lib64
        $ENV{CUDA_HOME}/lib64
        $ENV{CUDA_PATH}/lib64
)

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY
                 TensorRT_NVINFER_PLUGIN_LIBRARY
                 TensorRT_CUDART_LIBRARY TensorRT_CUBLAS_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS
        ${TensorRT_INCLUDE_DIR}
        ${TensorRT_INCLUDE_DIR}/../include  # sometimes NvInfer.h is in a subdir
    )
    set(TensorRT_LIBRARIES
        ${TensorRT_NVINFER_LIBRARY}
        ${TensorRT_NVINFER_PLUGIN_LIBRARY}
        ${TensorRT_CUDART_LIBRARY}
        ${TensorRT_CUBLAS_LIBRARY}
    )
    # Also find CUDA include dir if not separately specified
    find_path(CUDA_INCLUDE_DIR
        NAMES cuda_runtime.h
        PATHS
            /usr/include
            /usr/local/cuda/include
            $ENV{CUDA_DIR}/include
            $ENV{CUDA_HOME}/include
            $ENV{CUDA_PATH}/include
    )
    if(CUDA_INCLUDE_DIR)
        list(APPEND TensorRT_INCLUDE_DIRS ${CUDA_INCLUDE_DIR})
    endif()
    list(REMOVE_DUPLICATES TensorRT_INCLUDE_DIRS)

    get_filename_component(TensorRT_LIBRARY_DIRS ${TensorRT_NVINFER_LIBRARY} DIRECTORY)
endif()