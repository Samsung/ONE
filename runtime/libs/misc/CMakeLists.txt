# Library `nnfw_lib_misc`
file(GLOB_RECURSE NNFW_UTILITY_SRCS "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM NNFW_UTILITY_SRCS ${TESTS})

add_library(nnfw_lib_misc STATIC ${NNFW_UTILITY_SRCS})
target_include_directories(nnfw_lib_misc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nnfw_lib_misc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nnfw_lib_misc PRIVATE nnfw_common)
target_link_libraries(nnfw_lib_misc PRIVATE nnfw_coverage)

add_executable(nnfw_tensor_index_iterator "examples/tensor_index_iterator.cpp")
target_link_libraries(nnfw_tensor_index_iterator PRIVATE nnfw_lib_misc)

# Gtest misc_index_enumerator_test
if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(misc_index_enumerator_test src/IndexEnumerator.test.cpp)
target_link_libraries(misc_index_enumerator_test PRIVATE nnfw_lib_misc)

target_link_libraries(misc_index_enumerator_test PUBLIC gtest gtest_main ${LIB_PTHREAD})

add_test(misc_index_enumerator_test misc_index_enumerator_test)
install(TARGETS misc_index_enumerator_test DESTINATION unittest_standalone)
