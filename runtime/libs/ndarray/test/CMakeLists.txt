if(NOT BUILD_NDARRAY_TEST)
    return()
endif()

add_executable(ndarray_test ndarray_test.cpp)

target_link_libraries(ndarray_test PRIVATE ndarray)

nnfw_find_package(GTest)
if(NOT GTest_FOUND)
    message(STATUS "GTest not avaialble. Skipping NDArray test build")
    return()
endif(NOT GTest_FOUND)

target_link_libraries(ndarray_test PUBLIC gtest gtest_main ${LIB_PTHREAD})

add_test(ndarray_test ndarray_test)
install(TARGETS ndarray_test DESTINATION unittest_standalone)
