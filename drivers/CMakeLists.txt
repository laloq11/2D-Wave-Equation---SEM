# ── CPU executables ────────────────────────────────────────────────────────────

add_executable(openMP_integration cpu/openMP_integration.cpp)
target_link_libraries(openMP_integration PRIVATE spectralcm linalg culib OpenMP::OpenMP_CXX)
target_include_directories(openMP_integration PRIVATE ${PROJECT_SOURCE_DIR})

# ── CUDA executables ───────────────────────────────────────────────────────────

if(SPECTRAL_CUDA_AVAILABLE)

    add_executable(spectral01 cuda/spectral01.cu)
    target_link_libraries(spectral01 PRIVATE
        spectralcm_cuda spectralcm linalg
        CUDA::cudart CLI11::CLI11 OpenMP::OpenMP_CXX)
    target_include_directories(spectral01 PRIVATE ${PROJECT_SOURCE_DIR})

    add_executable(simple cuda/simple_linalg_example.cu)
    target_link_libraries(simple PRIVATE
        spectralcm_cuda CUDA::cudart CLI11::CLI11 OpenMP::OpenMP_CXX)
    target_include_directories(simple PRIVATE ${PROJECT_SOURCE_DIR})

    add_executable(hello cuda/hello.cu)
    target_link_libraries(hello PRIVATE
        spectralcm_cuda CUDA::cudart CLI11::CLI11 OpenMP::OpenMP_CXX)
    target_include_directories(hello PRIVATE ${PROJECT_SOURCE_DIR})

endif()
