if(NOT TARGET nnkit_support_tflite-1.7)
  return()
endif(NOT TARGET nnkit_support_tflite-1.7)

file(GLOB_RECURSE TESTS "*.test.cpp")
file(GLOB_RECURSE SOURCES "*.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nnsuite_conv_tflite SHARED ${SOURCES})
target_link_libraries(nnsuite_conv_tflite nnsuite_conv)
target_link_libraries(nnsuite_conv_tflite nnkit_support_tflite-1.7)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

add_executable(nnsuite_conv_tflite_test ${TESTS})
target_link_libraries(nnsuite_conv_tflite_test nnsuite_conv_tflite)
target_link_libraries(nnsuite_conv_tflite_test gtest_main)
add_test(nnsuite_conv_tflite_test nnsuite_conv_tflite_test)
