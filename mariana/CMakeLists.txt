cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)
IF (NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/Mariana" CACHE PATH "Mariana Installation Directory")
ENDIF()
if(NOT DEFINED MARIANA_VERSION_MAJOR)
  set(MARIANA_VERSION_MAJOR 1)
endif()
if(NOT DEFINED MARIANA_VERSION_MINOR)
  set(MARIANA_VERSION_MINOR 0)
endif()
if(NOT DEFINED MARIANA_VERSION_PATCH)
  set(MARIANA_VERSION_PATCH 0)
endif()
if(NOT DEFINED MARIANA_VERSION_BUILD)
  set(MARIANA_VERSION_BUILD 0)
endif()

project(Mariana VERSION ${MARIANA_VERSION_MAJOR}.${MARIANA_VERSION_MINOR}.${MARIANA_VERSION_PATCH}.${MARIANA_VERSION_BUILD} LANGUAGES CUDA  C CXX)

set(MARIANA_TARGET_VERSION ${CMAKE_PROJECT_VERSION} CACHE STRING "Mariana logical version")
set(MARIANA_TARGET_SOVERSION ${CMAKE_PROJECT_VERSION} CACHE STRING "Mariana soname version")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_CUDA "Compile with CUDA" OFF)
option(WITH_RKNN "Compile with RKNN" OFF)
option(WITH_TRT "Compile with TRT" OFF)

include(cmake/git.cmake)
include(cmake/instruction.cmake)
configure_file(version.h.in generated/version.h @ONLY)
configure_file(config.h.in generated/config.h @ONLY)

message(STATUS "Build info Git hash:${GIT_HASH} Version:${MARIANA_TARGET_VERSION} Timing:${COMPILE_TIME}")

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_HEADER_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})

IF("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
    message(STATUS "building for: debugging")
    set(CXX_FORMAT_SIGNEDNESS "-Wformat-signedness")
    set(CXX_FORMAT_SECURITY "-Werror=format-security")
    set(CXX_STACK_PROTECTOR "-fstack-protector-all")
    set(CXX_FLAGS_DEBUG "-O0")
    set(CMAKE_C_STANDARD 99)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -ggdb -Wall -Wextra -DNETDATA_INTERNAL_CHECKS=1 -DNETDATA_VERIFY_LOCKS=1 ${CXX_FORMAT_SIGNEDNESS} ${CXX_FORMAT_SECURITY} ${CXX_STACK_PROTECTOR} ${CXX_FLAGS_DEBUG}")
ELSE()
    message(STATUS "building for: release")
    cmake_policy(SET CMP0069 "NEW")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    IF(${ipo_supported})
        message(STATUS "link time optimization: supported")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    ELSE()
        message(STATUS "link time optimization: not supported")
    ENDIF()
ENDIF()
# if (!CMAKE_BUILD_TYPE STREQUAL "RELEASE")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -O3 -std=c++11 -lrt -ldl -ffast-math")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC -O3 -std=c++11 -lrt -ldl -ffast-math")
# endif()

include(cmake/abseil.cmake)
include(cmake/glog.cmake)
include(cmake/protobuf.cmake)
if (WITH_TRT)
  set(WITH_CUDA ON)
  add_definitions(-DWITH_TRT)
  include(cmake/cudnn.cmake)
  include(cmake/tensorRT.cmake)
endif()
if (WITH_CUDA)
  add_definitions(-DWITH_CUDA)
  include(cmake/cuda.cmake)
endif()
if (WITH_RKNN)
  add_definitions(-DWITH_RKNN)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/api)
include_directories(${CMAKE_BINARY_DIR}/generated)

include(${CMAKE_CURRENT_LIST_DIR}/core/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/structure/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/marc/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/maro/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/api/CMakeLists.txt)
list(APPEND MARIANA_SRC ${MARIANA_CORE_OBJECTS} ${MARIANA_MARC_OBJECTS}
  ${MARIANA_STRUCTURE_OBJECTS} ${MARIANA_MARO_OBJECTS})

add_library(${CMAKE_PROJECT_NAME}-static STATIC)
add_library(${CMAKE_PROJECT_NAME} SHARED)
target_sources(${CMAKE_PROJECT_NAME}-static PUBLIC ${MARIANA_SRC})
target_sources(${CMAKE_PROJECT_NAME} PUBLIC ${MARIANA_SRC})

target_link_libraries(${CMAKE_PROJECT_NAME}-static PUBLIC ${MARIANA_EXTERN_LIB})
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${MARIANA_EXTERN_LIB})

set_target_properties(${CMAKE_PROJECT_NAME}
  PROPERTIES
  VERSION ${MARIANA_TARGET_VERSION}
  SOVERSION ${MARIANA_TARGET_SOVERSION}
  OUTPUT_NAME "mariana"
  POSITION_INDEPENDENT_CODE 1
  PUBLIC_HEADER "${INSTALL_HEADERS}"
  DEBUG_POSTFIX "_d"
  )
set_target_properties(${CMAKE_PROJECT_NAME}-static
  PROPERTIES
  VERSION ${MARIANA_TARGET_VERSION}
  SOVERSION ${MARIANA_TARGET_SOVERSION}
  OUTPUT_NAME "mariana-static"
  POSITION_INDEPENDENT_CODE 1
  PUBLIC_HEADER "${INSTALL_HEADERS}"
  DEBUG_POSTFIX "_d"
  MACOSX_RPATH ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")

install(
  TARGETS
  ${CMAKE_PROJECT_NAME}-static
  ${CMAKE_PROJECT_NAME}
  ARCHIVE
  DESTINATION ${INSTALL_LIBDIR}
  COMPONENT lib
  RUNTIME
  DESTINATION ${INSTALL_BINDIR}
  COMPONENT bin
  LIBRARY
  DESTINATION ${INSTALL_LIBDIR}
  COMPONENT lib
  PUBLIC_HEADER
  DESTINATION ${INSTALL_INCLUDEDIR}
  COMPONENT dev
  )
