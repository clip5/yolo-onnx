# FindRKNN.cmake
# Locate Rockchip RKNN API library and headers.
#
# Sets:
#   RKNN_FOUND
#   RKNN_INCLUDE_DIRS
#   RKNN_LIBRARIES
#   RKNN_LIBRARY_DIRS
#
# Environment variables:
#   RKNN_DIR  (hint path)

find_path(RKNN_INCLUDE_DIR
    NAMES rknn_api.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/rknn/include
        /opt/rknpu/include
        /usr/include/librknn_api
        $ENV{RKNN_DIR}/include
        $ENV{RKNN_SDK_DIR}/include
        $ENV{RKNPU_DIR}/include
)

find_library(RKNN_LIBRARY
    NAMES rknn_api librknn_api
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/rknn/lib
        /opt/rknpu/lib
        /opt/rknpu/lib64
        $ENV{RKNN_DIR}/lib
        $ENV{RKNN_SDK_DIR}/lib
        $ENV{RKNPU_DIR}/lib
        $ENV{RKNPU_DIR}/lib64
)

mark_as_advanced(RKNN_INCLUDE_DIR RKNN_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RKNN
    REQUIRED_VARS RKNN_INCLUDE_DIR RKNN_LIBRARY
)

if(RKNN_FOUND)
    set(RKNN_INCLUDE_DIRS ${RKNN_INCLUDE_DIR})
    set(RKNN_LIBRARIES ${RKNN_LIBRARY})
    get_filename_component(RKNN_LIBRARY_DIRS ${RKNN_LIBRARY} DIRECTORY)
endif()