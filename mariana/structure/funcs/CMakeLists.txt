# include(${CMAKE_CURRENT_LIST_DIR}/cpu/CMakeLists.txt)
if(WITH_TRT)
  include(${CMAKE_CURRENT_LIST_DIR}/tensorRT/CMakeLists.txt)
endif()
if(WITH_RKNN)
  include(${CMAKE_CURRENT_LIST_DIR}/rknn/CMakeLists.txt)
endif()

file(GLOB CURRENT_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB CURRENT_HEAD_PATH ${CMAKE_CURRENT_LIST_DIR}/*.h)

set(LIB_NAME mariana_structure_funcs)
add_library(${LIB_NAME} OBJECT ${CURRENT_SOURCE_PATH})
target_sources(${LIB_NAME} PRIVATE ${CURRENT_SOURCE_PATH} PUBLIC ${CURRENT_HEAD_PATH})
list(APPEND MARIANA_STRUCTURE_OBJECTS $<TARGET_OBJECTS:${LIB_NAME}>)
