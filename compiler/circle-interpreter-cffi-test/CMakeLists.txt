if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv")
set(TEST_LIST_FILE "test.lst")

get_target_property(ARTIFACTS_PATH testDataGenerator BINARY_DIR)

add_test(
  NAME circle_interpreter_cffi_test
  COMMAND ${VIRTUALENV}/bin/python infer.py
                    --lib_path $<TARGET_FILE:circle_interpreter_cffi>
                    --test_list ${TEST_LIST_FILE}
                    --artifact_dir ${ARTIFACTS_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
