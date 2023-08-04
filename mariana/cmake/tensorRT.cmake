set(TENSORRT_ROOT ${CMAKE_SOURCE_DIR}/3rd_party/TensorRT-8.0.3.4)

set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)

link_directories(${TENSORRT_LIB_DIR})
include_directories(${TENSORRT_INCLUDE_DIR} ${TENSORRT_ROOT}/samples/common)

list(APPEND TRT_LIBRARY nvinfer)
list(APPEND TRT_LIBRARY nvonnxparser)

list(APPEND MARIANA_EXTERN_LIB ${TRT_LIBRARY})
