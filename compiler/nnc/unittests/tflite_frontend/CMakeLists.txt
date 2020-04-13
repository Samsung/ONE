file(GLOB_RECURSE TESTS "*.cpp")

#Feature detect:
execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_data/gen_test.py ${CMAKE_CURRENT_BINARY_DIR}
        OUTPUT_VARIABLE outp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE test_create_failed
)

if (NOT ${test_create_failed})
  add_definitions(-DTFLITE_TEST_DIR="${CMAKE_CURRENT_BINARY_DIR}/")
  nnc_add_unit_test(nnc_tflite_frontend_test ${TESTS})
  optional_target_link_libraries(nnc_tflite_frontend_test mir_tflite_importer mir)
endif()
