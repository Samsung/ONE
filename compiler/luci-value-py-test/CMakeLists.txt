if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv_2_12_1")
set(TEST_LIST_FILE "test.lst")

if(NOT CMAKE_CROSSCOMPILING)
  get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)
  add_test(NAME luci_value_py_test
      COMMAND ${VIRTUALENV}/bin/python -m pytest -sv test_luci_eval.py
              --test_list ${TEST_LIST_FILE}
              --artifacts ${ARTIFACTS_BIN_PATH}
              --luci_eval_driver $<TARGET_FILE:luci_eval_driver>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
else(NOT CMAKE_CROSSCOMPILING)
  # NOTE target test is carried out using reference input/output data from host
  #      test results. this is because it would be difficult to prepare
  #      TensorFlow lite for target device.
  #      thus, one must run the host test and then run the test in target device
  #      with the test result files from the host test.
  if(NOT DEFINED ENV{BUILD_HOST_EXEC})
    message(STATUS "BUILD_HOST_EXEC not set: Skip luci-value-py-test")
    return()
  endif(NOT DEFINED ENV{BUILD_HOST_EXEC})

  set(ARTIFACTS_BIN_PATH $ENV{BUILD_HOST_EXEC}/compiler/common-artifacts)
  add_test(NAME luci_value_py_test
      COMMAND ${VIRTUALENV}/bin/python -m pytest -sv test_luci_eval_ref.py
              --test_list ${TEST_LIST_FILE}
              --artifacts ${ARTIFACTS_BIN_PATH}
              --target_artifacts ${CMAKE_CURRENT_BINARY_DIR}
              --luci_eval_driver $<TARGET_FILE:luci_eval_driver>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif(NOT CMAKE_CROSSCOMPILING)
