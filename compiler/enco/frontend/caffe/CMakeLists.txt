nnas_find_package(CaffeProto QUIET)

if(NOT CaffeProto_FOUND)
  return()
endif(NOT CaffeProto_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(enco_caffe_frontend SHARED ${SOURCES})
target_include_directories(enco_caffe_frontend PRIVATE src)
target_link_libraries(enco_caffe_frontend coco_core)
target_link_libraries(enco_caffe_frontend coco_generic)
target_link_libraries(enco_caffe_frontend enco_intf_frontend)
target_link_libraries(enco_caffe_frontend enco_intf_cmdline)
target_link_libraries(enco_caffe_frontend morph)
target_link_libraries(enco_caffe_frontend caffeproto)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

nnas_find_package(Caffe QUIET)

if(NOT Caffe_FOUND)
  return()
endif(NOT Caffe_FOUND)

add_executable(enco_caffe_frontend_test ${TESTS})
target_include_directories(enco_caffe_frontend_test PRIVATE src)
target_link_libraries(enco_caffe_frontend_test gtest_main)
target_link_libraries(enco_caffe_frontend_test enco_caffe_frontend)
target_link_libraries(enco_caffe_frontend_test morph)
target_link_libraries(enco_caffe_frontend_test caffe)
add_test(enco_caffe_frontend_test enco_caffe_frontend_test)
