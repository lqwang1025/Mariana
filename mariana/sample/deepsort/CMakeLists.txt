cmake_minimum_required(VERSION 3.4.1)
SET(CMAKE_C_COMPILER   "arm-linux-gnueabihf-gcc")
SET(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
project(rknn_yolov8)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s -g -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -s -g -O3")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH lib64)
else()
  set(LIB_ARCH lib)
endif()

set(THIRED_PARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)
set(OpenCV_DIR  ${THIRED_PARTY_DIR}/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS opencv_world)
# 打印opencv的相关信息
message(STATUS "OpenCV library status: ${OpenCV_LIBRARIES}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# opencv
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

# rknn api
set(RKNN_API_PATH ${THIRED_PARTY_DIR}/librknn_api)

set(MAR_PATH ${THIRED_PARTY_DIR}/Mariana)

set(Eigen3_DIR ${THIRED_PARTY_DIR}/eigen3/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

include_directories(${MAR_PATH}/include)
set(MAR_LIB ${MAR_PATH}/lib/libmariana.so)
set(RKNN_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

set(CMAKE_INSTALL_RPATH "lib")

aux_source_directory(${CMAKE_SOURCE_DIR}/src DIR_SRCS)
add_executable(rknn_yolov8 ${DIR_SRCS})

target_link_libraries(rknn_yolov8
  ${MAR_LIB}
  ${RKNN_LIB}
  dl
  ${OpenCV_LIBS}
  )

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8)
install(TARGETS rknn_yolov8 DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(DIRECTORY input DESTINATION ./)
install(PROGRAMS ${RKNN_LIB} ${MAR_LIB} ${MAR_PATH}/lib/libmariana.so.1.0.0.0 ${THIRED_PARTY_DIR}/opencv/lib/libopencv_world.so.4.5 DESTINATION lib)
