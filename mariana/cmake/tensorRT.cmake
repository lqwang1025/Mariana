set(TENSORRT_ROOT ${CMAKE_SOURCE_DIR}/3rd_party/TensorRT)

set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)

link_directories(${TENSORRT_LIB_DIR})
include_directories(${TENSORRT_INCLUDE_DIR})

