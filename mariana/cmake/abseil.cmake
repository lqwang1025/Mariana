include(ExternalProject)

set(ABSEIL_ROOT ${CMAKE_SOURCE_DIR}/3rd_party/abseil-cpp)
set(ABSEIL_GIT_TAG 20211102.0)
set(ABSEIL_GIT_URL https://github.com/abseil/abseil-cpp)
set(ABSEIL_CONFIGURE cd ${ABSEIL_ROOT}/src/abseil-cpp && cmake -B build -D CMAKE_INSTALL_PREFIX=${ABSEIL_ROOT}
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_CXX_STANDARD=11 -D ABSL_PROPAGATE_CXX_STD=ON .)
set(ABSEIL_MAKE  cd ${ABSEIL_ROOT}/src/abseil-cpp/build && make -j8)

set(ABSEIL_INSTALL cd ${ABSEIL_ROOT}/src/abseil-cpp/build &&
  find ./ -name "*.o" | xargs ar cr libabsl.a && make install &&
  cd ${ABSEIL_ROOT}/lib && find ./ -name "*.a" | xargs -l rm &&
  mv ${ABSEIL_ROOT}/src/abseil-cpp/build/libabsl.a ./
)

ExternalProject_Add(abseil-cpp
  PREFIX            ${ABSEIL_ROOT}
  GIT_REPOSITORY    ${ABSEIL_GIT_URL}
  GIT_TAG           ${ABSEIL_GIT_TAG}
  CONFIGURE_COMMAND ${ABSEIL_CONFIGURE}
  BUILD_COMMAND     ${ABSEIL_MAKE}
  INSTALL_COMMAND   ${ABSEIL_INSTALL}
  ALWAYS FALSE
)

set(ABSEIL_LIB_DIR ${ABSEIL_ROOT}/lib)
set(ABSEIL_INCLUDE_DIR ${ABSEIL_ROOT}/include)
link_directories(${ABSEIL_LIB_DIR})
include_directories(${ABSEIL_INCLUDE_DIR})
