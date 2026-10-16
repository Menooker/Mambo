cmake_minimum_required(VERSION 3.24)

project(Mambo LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MAMBO_BUILD_CPP_TESTS "Build the C++ CUDA Driver API test tools" ON)
option(MAMBO_BUILD_REFERENCE "Build the CPU ggml reference tool" ON)
option(MAMBO_BUILD_PYTHON_TESTS "Register one pytest suite with CTest" ON)

enable_testing()

set(MAMBO_LLAMA_SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/llama.cpp" CACHE PATH "Pinned llama.cpp reference source")
set(MAMBO_LLAMA_REVISION "21b359d8c13e71b4d60ed1bed00eff5e59c640a2" CACHE STRING "Expected llama.cpp reference commit")
set(MAMBO_NLOHMANN_INCLUDE_DIR "${MAMBO_LLAMA_SOURCE_DIR}/vendor" CACHE PATH "Directory containing nlohmann/json.hpp")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(MAMBO_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/python" CACHE PATH "Mambo Python source directory")

if(MAMBO_BUILD_REFERENCE OR MAMBO_BUILD_CPP_TESTS)
    if(NOT EXISTS "${MAMBO_LLAMA_SOURCE_DIR}/vendor/nlohmann/json.hpp")
        message(FATAL_ERROR "Initialize the pinned llama.cpp submodule: git submodule update --init")
    endif()
endif()

if(MAMBO_BUILD_REFERENCE)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${MAMBO_LLAMA_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE _mambo_llama_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _mambo_llama_git_result
        ERROR_QUIET
    )
    if(NOT _mambo_llama_git_result EQUAL 0 OR NOT _mambo_llama_head STREQUAL MAMBO_LLAMA_REVISION)
        message(FATAL_ERROR "llama.cpp reference must be checked out at ${MAMBO_LLAMA_REVISION}; found ${_mambo_llama_head}")
    endif()

    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build the pinned ggml reference statically" FORCE)
    set(GGML_CPU ON CACHE BOOL "Enable the ggml CPU reference" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "No ggml GPU backend in the reference tool" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "Portable reference CPU build" FORCE)
    set(GGML_CCACHE OFF CACHE BOOL "Do not require ccache" FORCE)
    set(GGML_ALL_WARNINGS OFF CACHE BOOL "Disable third-party warning checks" FORCE)
    add_subdirectory("${MAMBO_LLAMA_SOURCE_DIR}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
    add_subdirectory(tests/cpp/reference)
endif()

if(MAMBO_BUILD_CPP_TESTS)
    add_subdirectory(tests/cpp/loadtest)
endif()

if(MAMBO_BUILD_PYTHON_TESTS)
    set(_mambo_pytest_env "PYTHONPATH=${MAMBO_PYTHON_SOURCE_DIR}")
    if(MAMBO_BUILD_CPP_TESTS)
        list(APPEND _mambo_pytest_env "MAMBO_LOADTEST_EXECUTABLE=$<TARGET_FILE:loadtest>")
    endif()
    if(MAMBO_BUILD_REFERENCE)
        list(APPEND _mambo_pytest_env
            "MAMBO_REFERENCE_EXECUTABLE=$<TARGET_FILE:mambo_ggml_reference>"
            "MAMBO_LLAMA_SOURCE_DIR=${MAMBO_LLAMA_SOURCE_DIR}")
    endif()
    add_test(NAME mambo_pytest
        COMMAND "${CMAKE_COMMAND}" -E env ${_mambo_pytest_env}
                "${Python3_EXECUTABLE}" -m pytest -q
                "${PROJECT_SOURCE_DIR}/tests/python")
endif()
