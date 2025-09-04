# Library `nnfw_lib_misc`
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nnfw_lib_misc STATIC ${SOURCES})
target_include_directories(nnfw_lib_misc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nnfw_lib_misc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nnfw_lib_misc PRIVATE nnfw_common)
target_link_libraries(nnfw_lib_misc PRIVATE nnfw_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(nnfw_lib_misc_test ${TESTS})
target_link_libraries(nnfw_lib_misc_test PRIVATE nnfw_lib_misc)
target_link_libraries(nnfw_lib_misc_test PRIVATE nnfw_common nnfw_coverage)
target_link_libraries(nnfw_lib_misc_test PUBLIC gtest gtest_main ${LIB_PTHREAD})

add_test(nnfw_lib_misc_test nnfw_lib_misc_test)
install(TARGETS nnfw_lib_misc_test DESTINATION unittest)
