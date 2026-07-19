# FindCANN.cmake
# Locate Huawei Ascend CANN (AscendCL) library and headers.
#
# Sets:
#   CANN_FOUND
#   CANN_INCLUDE_DIRS
#   CANN_LIBRARIES
#   CANN_LIBRARY_DIRS
#
# Environment variables:
#   ASCEND_DIR  or  CANN_DIR  (hint path)

find_path(CANN_INCLUDE_DIR
    NAMES acl/acl.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/Ascend/include
        /opt/Ascend/acllib/include
        /usr/local/Ascend/include
        $ENV{ASCEND_DIR}/include
        $ENV{ASCEND_HOME}/include
        $ENV{CANN_DIR}/include
        $ENV{ASCEND_TOOLKIT_HOME}/include
)

find_library(CANN_ACL_LIBRARY
    NAMES acl libacl
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/Ascend/lib64
        /opt/Ascend/acllib/lib64
        /usr/local/Ascend/lib64
        $ENV{ASCEND_DIR}/lib64
        $ENV{ASCEND_HOME}/lib64
        $ENV{CANN_DIR}/lib64
        $ENV{ASCEND_TOOLKIT_HOME}/lib64
)

find_library(CANN_DVPP_LIBRARY
    NAMES acl_dvpp libacl_dvpp
    PATHS
        /usr/lib64
        /usr/local/lib64
        /opt/Ascend/lib64
        /opt/Ascend/acllib/lib64
        /usr/local/Ascend/lib64
        $ENV{ASCEND_DIR}/lib64
        $ENV{ASCEND_HOME}/lib64
        $ENV{CANN_DIR}/lib64
        $ENV{ASCEND_TOOLKIT_HOME}/lib64
)

mark_as_advanced(CANN_INCLUDE_DIR CANN_ACL_LIBRARY CANN_DVPP_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CANN
    REQUIRED_VARS CANN_INCLUDE_DIR CANN_ACL_LIBRARY
)

if(CANN_FOUND)
    set(CANN_INCLUDE_DIRS ${CANN_INCLUDE_DIR})
    set(CANN_LIBRARIES
        ${CANN_ACL_LIBRARY}
    )
    if(CANN_DVPP_LIBRARY)
        list(APPEND CANN_LIBRARIES ${CANN_DVPP_LIBRARY})
    endif()
    get_filename_component(CANN_LIBRARY_DIRS ${CANN_ACL_LIBRARY} DIRECTORY)
endif()