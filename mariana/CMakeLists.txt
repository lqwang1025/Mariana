cmake_minimum_required(VERSION 3.21.0)
IF (NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Mariana Installation Directory")
ENDIF()
if(NOT DEFINED MARIANA_VERSION_MAJOR)
  set(MARIANA_VERSION_MAJOR 0)
endif()
if(NOT DEFINED MARIANA_VERSION_MINOR)
  set(MARIANA_VERSION_MINOR 0)
endif()
if(NOT DEFINED MARIANA_VERSION_PATCH)
  set(MARIANA_VERSION_PATCH 1)
endif()
if(NOT DEFINED MARIANA_VERSION_BUILD)
  set(MARIANA_VERSION_BUILD 5)
endif()

project(Mariana VERSION ${MARIANA_VERSION_MAJOR}.${MARIANA_VERSION_MINOR}.${MARIANA_VERSION_PATCH}.${MARIANA_VERSION_BUILD} LANGUAGES CUDA  C CXX)

set(MARIANA_TARGET_VERSION ${CMAKE_PROJECT_VERSION} CACHE STRING "Mariana logical version")
set(MARIANA_TARGET_SOVERSION ${CMAKE_PROJECT_VERSION} CACHE STRING "Mariana soname version")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(cmake/git.cmake)
include(cmake/instruction.cmake)
configure_file(version.h.in generated/version.h @ONLY)
configure_file(config.h.in generated/config.h @ONLY)

message(STATUS "Build info Git hash:${GIT_HASH} Version:${MARIANA_TARGET_VERSION}")

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_HEADER_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -g -O3 -std=c++11 -lrt -mavx -ldl -ffast-math")

include(cmake/abseil.cmake)
include(cmake/glog.cmake)
include(cmake/protobuf.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/generated)

include(${CMAKE_CURRENT_LIST_DIR}/core/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/structure/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/marc/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/maro/CMakeLists.txt)
list(APPEND MARIANA_SRC ${MARIANA_CORE_OBJECTS} ${MARIANA_MARC_OBJECTS}
  ${MARIANA_STRUCTURE_OBJECTS} ${MARIANA_MARO_OBJECTS})

add_library(${CMAKE_PROJECT_NAME}-static STATIC)
add_library(${CMAKE_PROJECT_NAME} SHARED)
target_sources(${CMAKE_PROJECT_NAME}-static PUBLIC ${MARIANA_SRC})
target_sources(${CMAKE_PROJECT_NAME} PUBLIC ${MARIANA_SRC})

target_link_libraries(${CMAKE_PROJECT_NAME}-static PUBLIC -lglog -labsl -lpthread -lprotobuf)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC -lglog -labsl -lpthread -lprotobuf)

set_target_properties(${CMAKE_PROJECT_NAME}-static PROPERTIES VERSION ${MARIANA_TARGET_VERSION} SOVERSION ${MARIANA_TARGET_SOVERSION})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES VERSION ${MARIANA_TARGET_VERSION} SOVERSION ${MARIANA_TARGET_SOVERSION})

set_target_properties(${CMAKE_PROJECT_NAME}-static PROPERTIES OUTPUT_NAME "mariana-static")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME "mariana")

add_subdirectory(app)
