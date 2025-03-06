# TODO run when test is enabled

# python3 venv folder
# NOTE Docker image for CI doesn't use venv
set(VENV_PATH "${CMAKE_SOURCE_DIR}/infra/overlay/venv")

# DEPEND FILES
set(FILE_DEPS_VALCHK )

macro(COPY_SCRIPT FILENAME)
  set(SCRIPT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
  set(SCRIPT_DST "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
  add_custom_command(
    OUTPUT ${SCRIPT_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${SCRIPT_SRC}" "${SCRIPT_DST}"
    DEPENDS ${SCRIPT_SRC}
    COMMENT "tools/onnx2circle: prepare ${FILENAME}"
  )
  list(APPEND FILE_DEPS_VALCHK "${SCRIPT_DST}")
endmacro(COPY_SCRIPT)

# copy test scripts
COPY_SCRIPT(run_value_test.sh)
COPY_SCRIPT(exec_onnx.py)
COPY_SCRIPT(exec_circle.py)
COPY_SCRIPT(make_circle_input.py)
COPY_SCRIPT(comp_onnx_circle.py)
COPY_SCRIPT(util_h5_file.py)
COPY_SCRIPT(util_validation.py)

# AddModel used in test.lst
set(UNIT_TEST_MODELS )

get_target_property(ONNX_ARTIFACTS_BIN_PATH gen_onnx_target BINARY_DIR)
get_target_property(CIRCLE_ARTIFACTS_BIN_PATH onnx2circle_models_target BINARY_DIR)

macro(AddModel MODEL_NAME)
  # copy ONNX to build folder
  set(TEST_ONNX_MODEL_SRC "${ONNX_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.onnx")
  set(TEST_ONNX_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.onnx")
  add_custom_command(
    OUTPUT ${TEST_ONNX_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_ONNX_MODEL_SRC}" "${TEST_ONNX_MODEL_DST}"
    DEPENDS ${TEST_ONNX_MODEL_SRC}
    COMMENT "onnx2circle-value-test: prepare ${MODEL_NAME}.onnx"
  )
  # copy Circle to build folder
  set(TEST_CIRCLE_MODEL_SRC "${CIRCLE_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.circle")
  set(TEST_CIRCLE_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_CIRCLE_MODEL_SRC}" "${TEST_CIRCLE_MODEL_DST}"
    DEPENDS ${TEST_CIRCLE_MODEL_SRC}
    COMMENT "onnx2circle-value-test: prepare ${MODEL_NAME}.circle"
  )

  list(APPEND UNIT_TEST_MODELS "${MODEL_NAME}")
  list(APPEND FILE_DEPS_VALCHK "${TEST_ONNX_MODEL_DST}")
  list(APPEND FILE_DEPS_VALCHK "${TEST_CIRCLE_MODEL_DST}")
endmacro(AddModel)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_custom_target(
  onnx2circle_value_test_target ALL
  DEPENDS onnx2circle_models_target ${FILE_DEPS_VALCHK}
)

foreach(MODEL IN ITEMS ${UNIT_TEST_MODELS})
  set(TEST_MODEL_BASE "${CMAKE_CURRENT_BINARY_DIR}/${MODEL}")
  add_test(
    NAME onnx2circle_value_test_${MODEL}
    COMMAND bash run_value_test.sh ${VENV_PATH} ${TEST_MODEL_BASE}
  )
endforeach()
