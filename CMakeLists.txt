cmake_minimum_required(VERSION 3.19)
project(gguf_logit_lens LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Internal tensor names and model layouts are revision-dependent.
set(LLAMA_REVISION 7951f601e89494f2ed12dbd4eb52e254b8401197)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_REVISION})
FetchContent_MakeAvailable(llama)
find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${llama_SOURCE_DIR}
    OUTPUT_VARIABLE LLAMA_ACTUAL_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLAMA_GIT_STATUS ERROR_QUIET)
if(NOT LLAMA_GIT_STATUS EQUAL 0 OR NOT LLAMA_ACTUAL_REVISION STREQUAL LLAMA_REVISION)
    message(FATAL_ERROR "llama.cpp checkout must be at ${LLAMA_REVISION}; internal model layouts are pinned")
endif()

add_executable(gguf-lens src/main.cpp)
target_include_directories(gguf-lens PRIVATE ${llama_SOURCE_DIR}/src ${llama_SOURCE_DIR}/vendor)
target_link_libraries(gguf-lens PRIVATE llama ggml-cpu)
target_compile_definitions(gguf-lens PRIVATE LENS_LLAMA_REVISION="${LLAMA_REVISION}")

enable_testing()
add_executable(test-metrics tests/test_metrics.cpp)
target_include_directories(test-metrics PRIVATE src)
add_test(NAME metrics COMMAND test-metrics)
add_test(NAME help COMMAND gguf-lens --help)
