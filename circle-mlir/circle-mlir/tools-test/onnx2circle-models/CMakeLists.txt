# TODO run when test is enabled

set(OPTION_O2C_SINGLE)
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug")
  set(OPTION_O2C_SINGLE "-o2c-single")
endif()

# DEPEND FILES
set(FILE_DEPS_VALCHK )

get_target_property(ONNX_ARTIFACTS_BIN_PATH gen_onnx_target BINARY_DIR)

macro(AddModel MODEL_NAME)
  # copy to build folder
  set(TEST_ONNX_MODEL_SRC "${ONNX_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.onnx")
  set(TEST_ONNX_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.onnx")
  add_custom_command(
    OUTPUT ${TEST_ONNX_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_ONNX_MODEL_SRC}" "${TEST_ONNX_MODEL_DST}"
    DEPENDS ${TEST_ONNX_MODEL_SRC}
    COMMENT "onnx2circle-models: prepare ${MODEL_NAME}.onnx"
  )

  set(TEST_CIRCLE_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST}
    COMMAND "$<TARGET_FILE:onnx2circle>" "${OPTION_O2C_SINGLE}" "${TEST_ONNX_MODEL_DST}" "${TEST_CIRCLE_MODEL_DST}"
    DEPENDS ${TEST_ONNX_MODEL_DST} onnx2circle
    COMMENT "onnx2circle-models: generate ${MODEL_NAME}.circle"
  )

  list(APPEND FILE_DEPS_VALCHK "${TEST_CIRCLE_MODEL_DST}")
endmacro(AddModel)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

# onnx2circle_models_target depends on onnx2circle tool and gen_onnx_target for onnx files
add_custom_target(
  onnx2circle_models_target ALL
  DEPENDS onnx2circle gen_onnx_target ${FILE_DEPS_VALCHK}
)

set(GEN_CIRCLE_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
