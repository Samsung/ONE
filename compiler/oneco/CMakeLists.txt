nnas_find_package(Protobuf QUIET)
nnas_find_package(ONNXSource EXACT 1.6.0 QUIET)

if(NOT Protobuf_FOUND)
  return()
endif(NOT Protobuf_FOUND)

if(NOT ONNXSource_FOUND)
  return()
endif(NOT ONNXSource_FOUND)

add_subdirectory(proto)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(moco_onnx_frontend SHARED ${SOURCES})
target_include_directories(moco_onnx_frontend PRIVATE src)
target_include_directories(moco_onnx_frontend PUBLIC include)
target_link_libraries(moco_onnx_frontend PUBLIC moco_onnx_proto)
target_link_libraries(moco_onnx_frontend PUBLIC loco)
target_link_libraries(moco_onnx_frontend PRIVATE stdex)
target_link_libraries(moco_onnx_frontend PRIVATE cwrap)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

add_executable(moco_onnx_frontend_test ${TESTS})
target_include_directories(moco_onnx_frontend_test PRIVATE src)
target_link_libraries(moco_onnx_frontend_test gtest_main)
target_link_libraries(moco_onnx_frontend_test moco_onnx_frontend)
add_test(moco_onnx_frontend_test moco_onnx_frontend_test)
