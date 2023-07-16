set(CUDNN_ROOT /usr/local/cuda)

set(CUDNN_LIB_DIR ${CUDNN_ROOT}/lib64)
set(CUDNN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

link_directories(${CUDNN_LIB_DIR})
include_directories(${CUDNN_INCLUDE_DIR})
