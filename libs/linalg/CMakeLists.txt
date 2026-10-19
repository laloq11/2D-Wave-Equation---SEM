set(LINALG_SOURCES
    src/linalg.cpp
)

add_library(linalg STATIC ${LINALG_SOURCES})
set_target_properties(linalg PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(linalg PUBLIC inc)

if(WIN32)
    install(TARGETS linalg RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    install(TARGETS linalg ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(BUILD_TESTING)
    add_executable(linalg_testrunner
        test/linalg_test.cpp
        test/main_test.cpp
    )
    target_link_libraries(linalg_testrunner PRIVATE linalg catch2_header)
    install(TARGETS linalg_testrunner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    add_test(NAME linalg_tests COMMAND linalg_testrunner)
endif()
