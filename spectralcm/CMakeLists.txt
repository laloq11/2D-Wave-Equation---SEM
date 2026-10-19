set(SPECTRALCM_SOURCES
    src/basisfunctions.cpp
    src/ontheFly.cpp
    src/params.cpp
    src/spectralcm.cpp
)

add_library(spectralcm STATIC ${SPECTRALCM_SOURCES})
set_target_properties(spectralcm PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(spectralcm PUBLIC inc)

target_link_libraries(spectralcm
    PUBLIC  linalg
            numeric
    PRIVATE OpenMP::OpenMP_CXX
)

if(SPECTRAL_CUDA_AVAILABLE)
    # Propagate cuda_spectral/inc and CUDA toolkit headers to all C++ consumers
    # so they can parse cuontheFly.cuh (included transitively via params.hpp).
    target_include_directories(spectralcm PUBLIC
        ${CMAKE_SOURCE_DIR}/cuda_spectral/inc
        ${CMAKE_SOURCE_DIR}/libs/culib/inc
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    # CUDA_ON propagates to all consumers (CXX and CUDA alike).
    # The CUDA keyword compat macros are scoped to CXX-only so that actual
    # CUDA compilation units (.cu) keep the real __device__/__global__ semantics.
    target_compile_definitions(spectralcm PUBLIC
        CUDA_ON
        $<$<COMPILE_LANGUAGE:CXX>:__device__=>
        $<$<COMPILE_LANGUAGE:CXX>:__host__=>
        $<$<COMPILE_LANGUAGE:CXX>:__global__=>
        $<$<COMPILE_LANGUAGE:CXX>:__shared__=>
        $<$<COMPILE_LANGUAGE:CXX>:__constant__=>
        "$<$<COMPILE_LANGUAGE:CXX>:__forceinline__=inline>"
    )
    target_link_libraries(spectralcm PRIVATE spectralcm_cuda)
endif()

if(WIN32)
    install(TARGETS spectralcm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    install(TARGETS spectralcm ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(BUILD_TESTING)
    add_executable(spectralcm_testrunner
        test/maintest.cpp
        test/test_ontheflyAlgorithm.cpp
        test/test_parameters.cpp
        test/test_shapefunctions.cpp
    )
    target_link_libraries(spectralcm_testrunner PRIVATE spectralcm catch2_header)
    install(TARGETS spectralcm_testrunner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    add_test(NAME spectralcm_tests COMMAND spectralcm_testrunner)
endif()
