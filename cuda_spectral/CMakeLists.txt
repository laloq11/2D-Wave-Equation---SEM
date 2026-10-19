set(SPECTRALCUDA_SOURCES
    src/cubase.cpp
    src/cuda_tests.cpp
    src/linalg_omp.cpp
    src/cuontheFly.cu
)

add_library(spectralcm_cuda STATIC ${SPECTRALCUDA_SOURCES})
set_target_properties(spectralcm_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(spectralcm_cuda PUBLIC inc)

target_link_libraries(spectralcm_cuda
    PUBLIC  spectralcm
            culib
    PRIVATE linalg
            CUDA::cudart
            OpenMP::OpenMP_CXX
)

if(WIN32)
    install(TARGETS spectralcm_cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    install(TARGETS spectralcm_cuda ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(BUILD_TESTING)
    add_executable(spectralcu_testrunner
        test/cutest_ontheflyAlgorithm.cpp
        test/cutest_onthefly.cu
        test/maintest.cpp
    )
    set_target_properties(spectralcu_testrunner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(spectralcu_testrunner PRIVATE
        spectralcm_cuda
        spectralcm
        linalg
        CUDA::cudart
        catch2_header
    )
    install(TARGETS spectralcu_testrunner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    add_test(NAME spectralcu_tests COMMAND spectralcu_testrunner)
endif()
