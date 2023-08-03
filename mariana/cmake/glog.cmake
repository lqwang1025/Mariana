include(ExternalProject)

set(GLOG_ROOT ${CMAKE_SOURCE_DIR}/3rd_party/glog)
set(GLOG_GIT_TAG  v0.6.0)
set(GLOG_GIT_URL https://github.com/google/glog)
set(GLOG_CONFIGURE cd ${GLOG_ROOT}/src/glog && cmake -B build -D CMAKE_INSTALL_PREFIX=${GLOG_ROOT}
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D BUILD_SHARED_LIBS=OFF -D WITH_GFLAGS=OFF -D WITH_GTEST=OFF -D WITH_UNWIND=OFF -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} .)
set(GLOG_MAKE  cd ${GLOG_ROOT}/src/glog/build && make -j8)
set(GLOG_INSTALL cd ${GLOG_ROOT}/src/glog/build && make install)

ExternalProject_Add(glog
  PREFIX            ${GLOG_ROOT}
  GIT_REPOSITORY    ${GLOG_GIT_URL}
  GIT_TAG           ${GLOG_GIT_TAG}
  CONFIGURE_COMMAND ${GLOG_CONFIGURE}
  BUILD_COMMAND     ${GLOG_MAKE}
  INSTALL_COMMAND   ${GLOG_INSTALL})

set(GLOG_LIB_DIR ${GLOG_ROOT}/lib)
set(GLOG_INCLUDE_DIR  ${GLOG_ROOT}/include)

link_directories(${GLOG_LIB_DIR})
include_directories(${GLOG_INCLUDE_DIR})
