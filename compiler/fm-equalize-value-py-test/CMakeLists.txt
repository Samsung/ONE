if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv")
set(TEST_LIST_FILE "test.lst")

if (NOT DEFINED ONE_PYTHON_VERSION_MINOR)
  message(STATUS "Warning: ONE_PYTHON_VERSION_MINOR not set")
  return()
endif()
set(PYTHON_SITE_PACKAGES "${VIRTUALENV}/lib/python3.${ONE_PYTHON_VERSION_MINOR}/site-packages")

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)
get_target_property(FM_EQUALIZE_BIN_PATH fm-equalize BINARY_DIR)
get_target_property(FME_DETECT_BIN_PATH fme-detect BINARY_DIR)
get_target_property(DALGONA_BIN_PATH dalgona BINARY_DIR)
get_target_property(FME_APPLY_BIN_PATH fme-apply BINARY_DIR)
set(FM_EQUALIZE_BIN "${FM_EQUALIZE_BIN_PATH}/fm-equalize")
set(FME_DETECT_BIN "${FME_DETECT_BIN_PATH}/fme-detect")
set(DALGONA_BIN "${DALGONA_BIN_PATH}/dalgona")
set(FME_APPLY_BIN "${FME_APPLY_BIN_PATH}/fme-apply")

macro(eval RECIPE)
  set(CIRCLE_FILE "${RECIPE}.circle")
  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${CIRCLE_FILE}")
  set(OPT_CIRCLE_FILE "${RECIPE}.opt.circle")
  set(OPT_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${OPT_CIRCLE_FILE}")

  # Run circle2circle for fusing instance normalization.
  add_custom_command(OUTPUT ${OPT_CIRCLE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:circle2circle> --fuse_instnorm ${CIRCLE_PATH} ${OPT_CIRCLE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:circle2circle> ${CIRCLE_PATH}
    COMMENT "Generate ${OPT_CIRCLE_FILE} for fusing instance normalization."
  )

  set(AFTER_CIRCLE_FILE "${RECIPE}.after.circle")
  set(AFETR_PATTERN_FILE "${RECIPE}.after.json")
  set(AFTER_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${AFTER_CIRCLE_FILE}")
  set(AFTER_CIRCLE_PATTERN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${AFETR_PATTERN_FILE}")

  # Apply fm-equalize
  add_custom_command(OUTPUT ${AFTER_CIRCLE_OUTPUT_PATH} ${AFTER_CIRCLE_PATTERN_PATH}
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PYTHON_SITE_PACKAGES}
            ${VIRTUALENV}/bin/python ${FM_EQUALIZE_BIN} -i ${OPT_CIRCLE_OUTPUT_PATH}
                                       -o ${AFTER_CIRCLE_OUTPUT_PATH} -f ${AFTER_CIRCLE_PATTERN_PATH}
                                       --fme_detect ${FME_DETECT_BIN} --dalgona ${DALGONA_BIN}
                                       --fme_apply ${FME_APPLY_BIN}
    DEPENDS ${FM_EQUALIZE_BIN} ${OPT_CIRCLE_OUTPUT_PATH} ${FME_APPLY_BIN} ${FME_DETECT_BIN} ${DALGONA_BIN}
    COMMENT "Apply fm-equalize to ${OPT_CIRCLE_OUTPUT_PATH} on ${PYTHON_SITE_PACKAGES}"
  )

  # depends
  list(APPEND TEST_DEPS ${AFTER_CIRCLE_OUTPUT_PATH} ${AFTER_CIRCLE_PATTERN_PATH})
endmacro(eval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_custom_target(fm_equalize_value_py_test_files ALL DEPENDS ${TEST_DEPS})
add_dependencies(fm_equalize_value_py_test_files common_artifacts_deps)

add_test(NAME fm_equalize_value_py_test
  COMMAND ${VIRTUALENV}/bin/python -m pytest -sv test_luci_eval.py
        --test_list ${TEST_LIST_FILE}
        --tflite_dir ${ARTIFACTS_BIN_PATH}
        --circle_dir ${CMAKE_CURRENT_BINARY_DIR}
        --luci_eval_driver $<TARGET_FILE:luci_eval_driver>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
