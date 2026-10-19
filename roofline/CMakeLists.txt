message(STATUS "Starting roofline build...")

# ── CPU (OpenMP) roofline ──────────────────────────────────────────────────────
set(target_name roofline_spectral)
add_library(${target_name} SHARED "")
target_sources(${target_name} PRIVATE openmp/ops.cpp)

target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${target_name}
    PUBLIC  culib
    PRIVATE OpenMP::OpenMP_CXX
            CLI11::CLI11
)

add_executable(roofline openmp/roofline_spectral.cpp)
target_include_directories(roofline PRIVATE inc)
target_link_libraries(roofline PRIVATE ${target_name} spectralcm linalg)

# ── CUDA roofline (single kernels) ────────────────────────────────────────────
if(SPECTRAL_CUDA_AVAILABLE)
    add_executable(roofline_cuda cuda/roofline.cu cuda/roofline_helpers.cu)
    target_link_libraries(roofline_cuda PRIVATE
        spectralcm_cuda spectralcm linalg CUDA::cudart CLI11::CLI11 culib)
    target_include_directories(roofline_cuda PRIVATE inc)

endif()

# ── Whole-simulation benchmark (OpenMP and CUDA devices) ──────────────────────
if(SPECTRAL_CUDA_AVAILABLE)
    add_executable(benchmark_solver solver/benchmark_solver.cu)
    target_link_libraries(benchmark_solver PRIVATE
        spectralcm_cuda spectralcm linalg CUDA::cudart CLI11::CLI11 culib)
    target_include_directories(benchmark_solver PRIVATE inc)
endif()
