set(NUMERIC_SOURCES
    src/GaussLobatto.cpp
    src/LagrangeBasisFunctions.cpp
)

add_library(numeric STATIC ${NUMERIC_SOURCES})
set_target_properties(numeric PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(numeric PUBLIC inc)

target_link_libraries(numeric PUBLIC linalg)

if(WIN32)
    install(TARGETS numeric RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    install(TARGETS numeric ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
