cmake_minimum_required(VERSION 3.18)
project(spectral-cm VERSION 0.1.0 LANGUAGES CXX)

# ============================================================
# Build options
# ============================================================
option(BUILD_TESTING        "Build unit tests"              ON)
option(BUILD_PYTHON         "Build Python bindings"         ON)
option(BUILD_BENCHMARKS     "Build roofline benchmarks"     ON)
option(SPECTRAL_USE_CUDA    "Enable CUDA GPU acceleration"  ON)

# ============================================================
# C++ standard
# ============================================================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_definitions(_INCLUDE_CSTDDEF)
endif()

# ============================================================
# Install paths
# ============================================================
include(GNUInstallDirs)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
        CACHE PATH "Default install path" FORCE)
endif()
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

# ============================================================
# Dependencies — discovered once, used everywhere
# ============================================================
find_package(OpenMP REQUIRED)
find_package(CLI11 REQUIRED)

if(SPECTRAL_USE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        # CUDA architecture: must be set before enable_language(CUDA).
        # Empty SCM_CUDA_ARCH = ask nvidia-smi for the local GPU(s).
        # Override with: cmake -DSCM_CUDA_ARCH="70;80" ..
        set(SCM_CUDA_ARCH "" CACHE STRING "CUDA architectures to build for (empty = detect the local GPU)")
        if(SCM_CUDA_ARCH)
            set(_cuda_arch "${SCM_CUDA_ARCH}")
        else()
            execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                            OUTPUT_VARIABLE _gpu_caps RESULT_VARIABLE _smi_result
                            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_smi_result EQUAL 0 AND _gpu_caps)
                # "5.0\n8.6" -> "50;86" (one entry per GPU, duplicates removed)
                string(REPLACE "." "" _gpu_caps "${_gpu_caps}")
                string(REGEX REPLACE "[\r\n]+" ";" _cuda_arch "${_gpu_caps}")
                list(REMOVE_DUPLICATES _cuda_arch)
            else()
                # no GPU visible (e.g. login node): build for every arch CUDA 12 supports
                set(_cuda_arch "50;52;60;61;70;75;80;86;89;90")
            endif()
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        enable_language(CUDA)
        set(SPECTRAL_CUDA_AVAILABLE TRUE)
        message(STATUS "CUDA ${CUDAToolkit_VERSION} found — building GPU targets")
    else()
        message(STATUS "CUDA not found — GPU targets will be skipped")
    endif()
endif()

# ============================================================
# Vendor headers
# ============================================================
add_library(catch2_header INTERFACE)
target_include_directories(catch2_header INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/catch)

# ============================================================
# Testing infrastructure
# ============================================================
if(BUILD_TESTING)
    enable_testing()
    include(CTest)
endif()

# ============================================================
# First-party libraries
# ============================================================
add_subdirectory(libs/linalg)
add_subdirectory(libs/numeric)
add_subdirectory(libs/culib)

# ============================================================
# Core library and optional GPU layer
# ============================================================
add_subdirectory(spectralcm)

if(SPECTRAL_CUDA_AVAILABLE)
    add_subdirectory(cuda_spectral)
endif()

# ============================================================
# Executables
# ============================================================
add_subdirectory(drivers)

# ============================================================
# Python bindings
# ============================================================
if(BUILD_PYTHON)
    set(PYBIND11_CPP_STANDARD -std=c++14)
    set(PYBIND11_INSTALL ON CACHE BOOL "Enable pybind11 install")
    add_subdirectory(vendor/pybind11)
    add_subdirectory(python)
endif()

# ============================================================
# Performance benchmarks (roofline)
# ============================================================
if(BUILD_BENCHMARKS)
    add_subdirectory(roofline)
endif()
