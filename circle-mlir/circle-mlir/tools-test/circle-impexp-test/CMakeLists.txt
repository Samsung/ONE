# TODO run when test is enabled

# python3 venv folder
# NOTE Docker image for CI doesn't use venv
set(VENV_PATH "${CMAKE_SOURCE_DIR}/infra/overlay/venv")

# DEPEND FILES
set(FILE_DEPS_VALCHK )
set(FILE_DEPS_IMP )

macro(COPY_SCRIPT FILENAME)
  set(SCRIPT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
  set(SCRIPT_DST "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
  add_custom_command(
    OUTPUT ${SCRIPT_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${SCRIPT_SRC}" "${SCRIPT_DST}"
    DEPENDS ${SCRIPT_SRC}
    COMMENT "circle-impexp-test: prepare ${FILENAME}"
  )
  list(APPEND FILE_DEPS_VALCHK "${SCRIPT_DST}")
endmacro(COPY_SCRIPT)

# copy test scripts
COPY_SCRIPT(run_value_test.sh)
COPY_SCRIPT(run_import_test.sh)
COPY_SCRIPT(exec_onnx.py)
COPY_SCRIPT(exec_circle.py)
COPY_SCRIPT(make_circle_input.py)
COPY_SCRIPT(comp_circle_circle.py)
COPY_SCRIPT(util_h5_file.py)

get_target_property(ONNX_ARTIFACTS_BIN_PATH gen_onnx_target BINARY_DIR)
get_target_property(CIRCLE_ARTIFACTS_BIN_PATH onnx2circle_models_target BINARY_DIR)

# AddModel used in test.lst
set(UNIT_TEST_MODELS )
# ImportModel used in test.lst
set(UNIT_TEST_MODELS_IMPORT )

macro(AddModel MODEL_NAME)
  # copy ONNX to test folder for generation of I/O data
  set(TEST_ONNX_MODEL_SRC "${ONNX_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.onnx")
  set(TEST_ONNX_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.onnx")
  add_custom_command(
    OUTPUT ${TEST_ONNX_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_ONNX_MODEL_SRC}" "${TEST_ONNX_MODEL_DST}"
    DEPENDS ${TEST_ONNX_MODEL_SRC}
    COMMENT "circle-impexp-test: copy ${MODEL_NAME}.onnx"
  )

  # copy circle to test folder from value test folder as first circle model
  set(TEST_CIRCLE_MODEL_SRC "${CIRCLE_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.circle")
  set(TEST_CIRCLE_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_CIRCLE_MODEL_SRC}" "${TEST_CIRCLE_MODEL_DST}"
    DEPENDS ${TEST_CIRCLE_MODEL_SRC} onnx2circle_models_target
    COMMENT "circle-impexp-test: copy ${MODEL_NAME}.circle"
  )

  # run circle_impexp tool to generate second circle file
  set(TEST_CIRCLE_MODEL_DST2 "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.2.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST2}
    COMMAND "$<TARGET_FILE:circle_impexp>" "${TEST_CIRCLE_MODEL_DST}" "${TEST_CIRCLE_MODEL_DST2}"
    DEPENDS ${TEST_CIRCLE_MODEL_DST} circle_impexp
    COMMENT "circle-impexp-test: generate ${MODEL_NAME}.2.circle"
  )

  list(APPEND UNIT_TEST_MODELS "${MODEL_NAME}")
  list(APPEND FILE_DEPS_VALCHK "${TEST_ONNX_MODEL_DST}")
  list(APPEND FILE_DEPS_VALCHK "${TEST_CIRCLE_MODEL_DST}")
  list(APPEND FILE_DEPS_VALCHK "${TEST_CIRCLE_MODEL_DST2}")

endmacro(AddModel)

macro(ImportModel MODEL_NAME)
  # copy circle to test folder from value test folder as first circle model
  set(TEST_CIRCLE_MODEL_SRC "${CIRCLE_ARTIFACTS_BIN_PATH}/${MODEL_NAME}.circle")
  set(TEST_CIRCLE_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_CIRCLE_MODEL_SRC}" "${TEST_CIRCLE_MODEL_DST}"
    DEPENDS ${TEST_CIRCLE_MODEL_SRC} onnx2circle_models_target
    COMMENT "circle-impexp-test: copy ${MODEL_NAME}.circle"
  )

  # run circle_impexp tool to generate second circle file
  set(TEST_CIRCLE_MODEL_DST2 "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}.2.circle")
  add_custom_command(
    OUTPUT ${TEST_CIRCLE_MODEL_DST2}
    COMMAND "$<TARGET_FILE:circle_impexp>" "${TEST_CIRCLE_MODEL_DST}" "${TEST_CIRCLE_MODEL_DST2}"
    DEPENDS circle_impexp ${TEST_CIRCLE_MODEL_DST}
    COMMENT "circle-impexp-test: generate ${MODEL_NAME}.2.circle"
  )

  list(APPEND UNIT_TEST_MODELS_IMPORT "${MODEL_NAME}")
  list(APPEND FILE_DEPS_IMP "${TEST_CIRCLE_MODEL_DST}")
  list(APPEND FILE_DEPS_IMP "${TEST_CIRCLE_MODEL_DST2}")

endmacro(ImportModel)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_custom_target(
  circle_impexp_value_test_target ALL
  DEPENDS onnx2circle_value_test_target ${FILE_DEPS_VALCHK}
)

foreach(MODEL IN ITEMS ${UNIT_TEST_MODELS})
  set(TEST_MODEL_BASE "${CMAKE_CURRENT_BINARY_DIR}/${MODEL}")
  add_test(
    NAME circle_impexp_value_test_${MODEL}
    COMMAND bash run_value_test.sh ${VENV_PATH} ${TEST_MODEL_BASE}
  )
endforeach()

add_custom_target(
  circle_impexp_import_test_target ALL
  DEPENDS ${FILE_DEPS_IMP}
)

foreach(MODEL IN ITEMS ${UNIT_TEST_MODELS_IMPORT})
  set(TEST_MODEL_BASE "${CMAKE_CURRENT_BINARY_DIR}/${MODEL}")
  add_test(
    NAME circle_impexp_import_test_${MODEL}
    COMMAND bash run_import_test.sh ${TEST_MODEL_BASE}
  )
endforeach()
