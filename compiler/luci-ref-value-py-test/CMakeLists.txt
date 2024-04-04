if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv_2_12_1")
set(TEST_LIST_FILE "test.lst")

nncc_find_resource(TensorFlowLiteRecipes)
set(TFLITE_RECIPE_REPO "${TensorFlowLiteRecipes_DIR}")

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)
add_test(NAME luci_ref_value_py_test
    COMMAND ${VIRTUALENV}/bin/python -m pytest -sv test_luci_eval.py
            --test_list ${TEST_LIST_FILE}
            --tflrecipe ${TFLITE_RECIPE_REPO}
            --artifacts ${ARTIFACTS_BIN_PATH}
            --binary ${CMAKE_CURRENT_BINARY_DIR}
            --luci_eval_driver $<TARGET_FILE:luci_eval_driver>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
