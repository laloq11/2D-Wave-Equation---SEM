pybind11_add_module(pyspectralcm bindings/spectralcm_python.cpp)
target_link_libraries(pyspectralcm PRIVATE spectralcm)

if(SPECTRAL_CUDA_AVAILABLE)
    target_link_libraries(pyspectralcm PRIVATE spectralcm_cuda)
endif()

install(TARGETS pyspectralcm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

file(GLOB PYTHON_SCRIPTS CONFIGURE_DEPENDS scripts/*.py)
foreach(filename ${PYTHON_SCRIPTS})
    install(FILES ${filename} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endforeach()
