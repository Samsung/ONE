nnas_find_package(FlatBuffers EXACT 23.5.26 QUIET)

if(NOT FlatBuffers_FOUND)
  return()
endif(NOT FlatBuffers_FOUND)

FlatBuffers_Target(enco_tflite_schema
                   OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
                   SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schema"
                   SCHEMA_FILES schema.fbs)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(enco_tflite_frontend SHARED ${SOURCES})
target_include_directories(enco_tflite_frontend PRIVATE src)
target_link_libraries(enco_tflite_frontend enco_intf_frontend)
target_link_libraries(enco_tflite_frontend enco_intf_cmdline)
target_link_libraries(enco_tflite_frontend enco_tflite_schema)
target_link_libraries(enco_tflite_frontend morph)
target_link_libraries(enco_tflite_frontend cwrap)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

add_executable(enco_tflite_frontend_test ${TESTS})
target_include_directories(enco_tflite_frontend_test PRIVATE src)
target_link_libraries(enco_tflite_frontend_test gtest_main)
target_link_libraries(enco_tflite_frontend_test enco_tflite_frontend)
add_test(enco_tflite_frontend_test enco_tflite_frontend_test)
