include(${CMAKE_CURRENT_LIST_DIR}/utils/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/impl/CMakeLists.txt)

if(WITH_CUDA)
  include(${CMAKE_CURRENT_LIST_DIR}/cuda/CMakeLists.txt)
endif()

file(GLOB CURRENT_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB CURRENT_HEAD_PATH ${CMAKE_CURRENT_LIST_DIR}/*.h)

set(LIB_NAME mariana_core)
add_library(${LIB_NAME} OBJECT ${CURRENT_SOURCE_PATH})
target_sources(${LIB_NAME} PRIVATE ${CURRENT_SOURCE_PATH} PUBLIC ${CURRENT_HEAD_PATH})
add_dependencies(${LIB_NAME} abseil-cpp glog)
list(APPEND MARIANA_CORE_OBJECTS $<TARGET_OBJECTS:${LIB_NAME}>)
