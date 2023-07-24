set(GIT_HASH "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR})
endif()
string(TIMESTAMP COMPILE_TIME %Y-%m-%d_%H:%M:%S)
