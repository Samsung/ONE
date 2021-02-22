if(NOT BUILD_LIB_TFLITE_V2)
  return()
endif(NOT BUILD_LIB_TFLITE_V2)

nnfw_find_package(TensorFlowLite EXACT 2.3.0 QUIET)
if(NOT TensorFlowLite_2_3_0_FOUND)
  message(STATUS "Check tensorflow lite 2.3.0 library extension build: need tensorflow lite 2.3.0 library")
  return()
endif(NOT TensorFlowLite_2_3_0_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nnfw_lib_tflite_v2 STATIC ${SOURCES})
set_target_properties(nnfw_lib_tflite_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_lib_tflite_v2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nnfw_lib_tflite_v2 PUBLIC tensorflow-lite-2.3.0)
target_link_libraries(nnfw_lib_tflite_v2 PUBLIC nnfw_lib_misc)
target_link_libraries(nnfw_lib_tflite_v2 PRIVATE ${LIB_PTHREAD} dl)
target_link_libraries(nnfw_lib_tflite_v2 PRIVATE nnfw_common)
target_link_libraries(nnfw_lib_tflite_v2 PRIVATE nnfw_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(nnfw_lib_tflite_v2_test_TensorView src/TensorView.test.cpp)
target_link_libraries(nnfw_lib_tflite_v2_test_TensorView nnfw_lib_tflite_v2)
