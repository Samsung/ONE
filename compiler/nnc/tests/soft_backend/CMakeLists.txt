file(GLOB_RECURSE SOFT_TEST_DEF_SOURCES *.def)

nnc_make_generated_sources("${SOFT_TEST_DEF_SOURCES}" ${CMAKE_CURRENT_BINARY_DIR} SOFT_TEST_GENERATED_SOURCES)

add_executable(nnc_system_soft_backend_cpp_compile ${SOFT_TEST_GENERATED_SOURCES} CompileCPP.cpp)
target_link_libraries(nnc_system_soft_backend_cpp_compile PRIVATE soft_backend_cpp mir)
target_include_directories(nnc_system_soft_backend_cpp_compile PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                                                       ${NNC_SOFT_BACKEND_DIR})

add_test(nnc_system_test_soft_backend_cpp_compile nnc_system_soft_backend_cpp_compile)
