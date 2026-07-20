# FindONNXRuntime.cmake
# Locate ONNX Runtime library and headers.
#
# Sets:
#   ONNXRUNTIME_FOUND
#   ONNXRUNTIME_INCLUDE_DIRS
#   ONNXRUNTIME_LIBRARIES
#   ONNXRUNTIME_LIBRARY_DIRS

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/onnxruntime/include
        $ENV{ONNXRUNTIME_DIR}/include
        $ENV{HOME}/download/onnxruntime/include
        $ENV{HOME}/.cache/.lingma/env/include
    PATH_SUFFIXES
        onnxruntime/core/session    # conda package layout
        core/session                # alternate layout
)

# If not found at root, check under common subdirectories
if(NOT ONNXRUNTIME_INCLUDE_DIR)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            /usr/include/onnxruntime/core/session
            /usr/local/include/onnxruntime/core/session
            /opt/onnxruntime/include/onnxruntime/core/session
            $ENV{ONNXRUNTIME_DIR}/include/onnxruntime/core/session
    )
endif()

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/onnxruntime/lib
        $ENV{ONNXRUNTIME_DIR}/lib
        $ENV{HOME}/download/onnxruntime/lib
        $ENV{HOME}/.cache/.lingma/env
)

find_library(ONNXRUNTIME_PROVIDERS_LIBRARY
    NAMES onnxruntime_providers_shared libonnxruntime_providers_shared
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/onnxruntime/lib
        $ENV{ONNXRUNTIME_DIR}/lib
        $ENV{HOME}/download/onnxruntime/lib
        $ENV{HOME}/.cache/.lingma/env
)

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
)

if(ONNXRUNTIME_FOUND)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
    if(ONNXRUNTIME_PROVIDERS_LIBRARY)
        list(APPEND ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_PROVIDERS_LIBRARY})
    endif()
    get_filename_component(ONNXRUNTIME_LIBRARY_DIRS ${ONNXRUNTIME_LIBRARY} DIRECTORY)
endif()