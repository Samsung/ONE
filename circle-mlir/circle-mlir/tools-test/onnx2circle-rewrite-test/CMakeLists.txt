# TODO run when test is enabled

# python3 venv folder
# NOTE Docker image for CI doesn't use venv
set(VENV_PATH "${CMAKE_SOURCE_DIR}/infra/overlay/venv")

set(OPTION_O2C_SINGLE)
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug")
  set(OPTION_O2C_SINGLE "-o2c-single")
endif()

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
COPY_SCRIPT(run_circle_ops_test.sh)
COPY_SCRIPT(check_circle_ops.py)

# AddModel used in test.lst
set(UNIT_TEST_MODELS )

get_target_property(ONNX_ARTIFACTS_BIN_PATH gen_onnx_target BINARY_DIR)

macro(AddModel MODEL_NAME)
  # copy ONNX to build folder
  set(TEST_ONNX_MODEL_SRC "${ONNX_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.onnx")
  set(TEST_ONNX_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.onnx")
  add_custom_command(
    OUTPUT ${TEST_ONNX_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_ONNX_MODEL_SRC}" "${TEST_ONNX_MODEL_DST}"
    DEPENDS ${TEST_ONNX_MODEL_SRC}
    COMMENT "onnx2circle-rewrite-test: prepare ${MODEL_NAME}.onnx"
  )
  # generate .ops files containing Operators
  set(TEST_CIRCLE_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST}.ops
    COMMAND "$<TARGET_FILE:onnx2circle>" --save_ops "${OPTION_O2C_SINGLE}"
            "${TEST_ONNX_MODEL_DST}" "${TEST_CIRCLE_MODEL_DST}"
    DEPENDS ${TEST_ONNX_MODEL_DST} onnx2circle
    COMMENT "onnx2circle-rewrite-test: generate ${MODEL_NAME}.circle.ops"
  )

  list(APPEND UNIT_TEST_MODELS "${MODEL_NAME}")
  list(APPEND FILE_DEPS_VALCHK "${TEST_CIRCLE_MODEL_DST}.ops")
endmacro(AddModel)

# Read "test.lst"
include("test.lst")

add_custom_target(
  onnx2circle_rewrite_test_target ALL
  DEPENDS onnx2circle_models_target ${FILE_DEPS_VALCHK}
)

set(MODELS_ROOT "${CMAKE_SOURCE_DIR}/models")
set(RUN_PATH "${CMAKE_CURRENT_BINARY_DIR}")

foreach(MODEL IN ITEMS ${UNIT_TEST_MODELS})
  add_test(
    NAME onnx2circle_circle_ops_${MODEL}
    COMMAND bash run_circle_ops_test.sh ${VENV_PATH} ${RUN_PATH} ${MODELS_ROOT} ${MODEL}
  )
endforeach()
