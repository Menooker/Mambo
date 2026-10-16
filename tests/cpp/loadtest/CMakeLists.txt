file(GLOB _cuda_python_include_candidates
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/*/include"
)
if(Python3_EXECUTABLE)
    get_filename_component(_mambo_python_bin_dir "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(_mambo_python_env_root "${_mambo_python_bin_dir}" DIRECTORY)
    file(GLOB _mambo_cuda_python_include_candidates
        "${_mambo_python_env_root}/lib/python*/site-packages/nvidia/*/include"
    )
    list(APPEND _cuda_python_include_candidates ${_mambo_cuda_python_include_candidates})
endif()

find_path(CUDA_DRIVER_INCLUDE_DIR
    NAMES cuda.h
    HINTS
        "$ENV{CONDA_PREFIX}/include"
        ${_cuda_python_include_candidates}
)

find_library(CUDA_DRIVER_LIBRARY
    NAMES cuda
    HINTS
        "$ENV{CONDA_PREFIX}/lib"
)

if(NOT CUDA_DRIVER_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR
        "CUDA Driver API development files were not found. Activate the mambo "
        "conda environment or install the CUDA driver development package."
    )
endif()

if(NOT MAMBO_NLOHMANN_INCLUDE_DIR OR
   NOT EXISTS "${MAMBO_NLOHMANN_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR
        "Set MAMBO_NLOHMANN_INCLUDE_DIR to the directory containing nlohmann/json.hpp "
        "(the local llama.cpp vendor directory is suitable)."
    )
endif()

add_executable(loadtest loadtest.cpp)
target_compile_features(loadtest PRIVATE cxx_std_17)
target_include_directories(loadtest PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}" "${MAMBO_NLOHMANN_INCLUDE_DIR}")
target_link_libraries(loadtest PRIVATE "${CUDA_DRIVER_LIBRARY}")
