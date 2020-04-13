file(GLOB_RECURSE TESTS "*.cpp")

execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_data/gen_model.py ${CMAKE_CURRENT_BINARY_DIR}
        OUTPUT_VARIABLE outp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE test_create_failed
)

if (NOT ${test_create_failed})
  add_definitions(-DTEST_DIR="${CMAKE_CURRENT_BINARY_DIR}")
  nnc_add_unit_test(nnc_caffe2_frontend_test ${TESTS})
  optional_target_link_libraries(nnc_caffe2_frontend_test mir_caffe2_importer mir)
endif()
