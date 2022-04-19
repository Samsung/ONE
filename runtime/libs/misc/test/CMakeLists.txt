if(NOT TARGET nnfw_lib_misc)
    return()
endif()

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(nnfw_index_enumerator_test IndexEnumerator.test.cpp)
target_link_libraries(nnfw_index_enumerator_test PRIVATE nnfw_lib_misc)
target_link_libraries(nnfw_index_enumerator_test PRIVATE nnfw_coverage)
target_link_libraries(nnfw_index_enumerator_test PUBLIC gtest gtest_main ${LIB_PTHREAD})

add_test(nnfw_index_enumerator_test nnfw_index_enumerator_test)
install(TARGETS nnfw_index_enumerator_test DESTINATION unittest_standalone)
