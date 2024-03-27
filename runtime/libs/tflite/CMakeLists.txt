nnfw_find_package(TensorFlowLite EXACT 2.15.0 QUIET)
if(NOT TensorFlowLite_FOUND)
  message(STATUS "Check tensorflow lite library extension build: need tensorflow lite library")
  return()
endif(NOT TensorFlowLite_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nnfw_lib_tflite STATIC ${SOURCES})
set_target_properties(nnfw_lib_tflite PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_lib_tflite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nnfw_lib_tflite PUBLIC tensorflow-lite-2.15.0)
target_link_libraries(nnfw_lib_tflite PUBLIC nnfw_lib_misc)
target_link_libraries(nnfw_lib_tflite PRIVATE ${LIB_PTHREAD} dl)
target_link_libraries(nnfw_lib_tflite PRIVATE nnfw_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(nnfw_lib_tflite_test_TensorView src/TensorView.test.cpp)
target_link_libraries(nnfw_lib_tflite_test_TensorView nnfw_lib_tflite)
