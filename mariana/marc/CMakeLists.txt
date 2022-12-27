include(${CMAKE_CURRENT_LIST_DIR}/onnx/CMakeLists.txt)

set(LIB_NAME mariana_marc)
add_library(${LIB_NAME}-static STATIC "")
add_library(${LIB_NAME} SHARED "")
file(GLOB CURRENT_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB CURRENT_HEAD_PATH ${CMAKE_CURRENT_LIST_DIR}/*.h)

target_sources(${LIB_NAME} PRIVATE ${CURRENT_SOURCE_PATH} ${MARIANA_MARC_OBJECTS} PUBLIC ${CURRENT_HEAD_PATH})
add_dependencies(${LIB_NAME} abseil-cpp glog protobuf)
target_link_libraries(${LIB_NAME} -lglog -labsl -lpthread)

target_sources(${LIB_NAME}-static PRIVATE ${CURRENT_SOURCE_PATH} ${MARIANA_MARC_OBJECTS} PUBLIC ${CURRENT_HEAD_PATH})
add_dependencies(${LIB_NAME}-static abseil-cpp glog protobuf)
target_link_libraries(${LIB_NAME}-static -lglog -labsl -lpthread -lprotobuf)
