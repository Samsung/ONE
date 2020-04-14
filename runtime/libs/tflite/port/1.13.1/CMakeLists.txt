if(NOT SUPPORT_TFLITE_VERSION VERSION_EQUAL 1.13.1)
  return()
endif(NOT SUPPORT_TFLITE_VERSION VERSION_EQUAL 1.13.1)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(tensorflow-lite-ex STATIC ${SOURCES})
set_target_properties(tensorflow-lite-ex PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tensorflow-lite-ex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tensorflow-lite-ex PUBLIC tensorflow-lite)
target_link_libraries(tensorflow-lite-ex PUBLIC nnfw_lib_misc nnfw_lib_rua_shim)
target_link_libraries(tensorflow-lite-ex PRIVATE ${LIB_PTHREAD} dl)
target_link_libraries(tensorflow-lite-ex PRIVATE nnfw_common)
target_link_libraries(tensorflow-lite-ex PRIVATE nnfw_coverage)
