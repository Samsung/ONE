if(NOT TARGET nnkit_support_caffe)
  return()
endif(NOT TARGET nnkit_support_caffe)

file(GLOB_RECURSE TESTS "*.test.cpp")
file(GLOB_RECURSE SOURCES "*.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nnsuite_conv_caffe SHARED ${SOURCES})
target_link_libraries(nnsuite_conv_caffe nnsuite_conv)
target_link_libraries(nnsuite_conv_caffe nnkit_support_caffe)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

add_executable(nnsuite_conv_caffe_test ${TESTS})
target_link_libraries(nnsuite_conv_caffe_test nnsuite_conv_caffe)
target_link_libraries(nnsuite_conv_caffe_test morph)
target_link_libraries(nnsuite_conv_caffe_test gtest_main)
add_test(nnsuite_conv_caffe_test nnsuite_conv_caffe_test)
