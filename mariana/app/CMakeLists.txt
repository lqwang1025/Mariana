project(App)

link_directories(${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

add_executable(app)
add_dependencies(app ${CMAKE_PROJECT_NAME}-static ${CMAKE_PROJECT_NAME})
file(GLOB CURRENT_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB CURRENT_HEAD_PATH ${CMAKE_CURRENT_LIST_DIR}/*.h)
target_sources(app PRIVATE ${CURRENT_SOURCE_PATH} PUBLIC ${CURRENT_HEAD_PATH})
target_link_libraries(app -Wl,-whole-archive libmariana-static.a -Wl,-no-whole-archive -lglog -labsl -lpthread -lprotobuf)
