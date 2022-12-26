include(ExternalProject)

set(PROTOBUF_ROOT ${CMAKE_SOURCE_DIR}/3rd_party/protobuf)
set(PROTOBUF_GIT_TAG  v21.10)
set(PROTOBUF_GIT_URL https://github.com/protocolbuffers/protobuf)
set(PROTOBUF_CONFIGURE cd ${PROTOBUF_ROOT}/src/protobuf && cmake -B build -D CMAKE_INSTALL_PREFIX=${PROTOBUF_ROOT}
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D protobuf_BUILD_TESTS=OFF -D BUILD_SHARED_LIBS=OFF -D protobuf_BUILD_TESTS=OFF.)
set(PROTOBUF_MAKE  cd ${PROTOBUF_ROOT}/src/protobuf/build && make -j8)
set(PROTOBUF_INSTALL cd ${PROTOBUF_ROOT}/src/protobuf/build && make install)

ExternalProject_Add(protobuf
  PREFIX            ${PROTOBUF_ROOT}
  GIT_REPOSITORY    ${PROTOBUF_GIT_URL}
  GIT_TAG           ${PROTOBUF_GIT_TAG}
  CONFIGURE_COMMAND ${PROTOBUF_CONFIGURE}
  BUILD_COMMAND     ${PROTOBUF_MAKE}
  INSTALL_COMMAND   ${PROTOBUF_INSTALL})

set(PROTOBUF_LIB_DIR ${PROTOBUF_ROOT}/lib)
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/include)
set(PROTOBUF_BIN_DIR ${PROTOBUF_ROOT}/bin)

link_directories(${PROTOBUF_LIB_DIR})
include_directories(${PROTOBUF_INCLUDE_DIR})
