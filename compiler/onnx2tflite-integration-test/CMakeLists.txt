nnas_include(TargetRequire)
nncc_find_resource(ONNXTests)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS onnx2tflite)
list(APPEND REQUIRED_TARGETS onnxkit)
list(APPEND REQUIRED_TARGETS nnkit-run)
list(APPEND REQUIRED_TARGETS nnkit_onnx_backend)
list(APPEND REQUIRED_TARGETS nnkit_tflite_backend)
list(APPEND REQUIRED_TARGETS nnkit_randomize_action)
list(APPEND REQUIRED_TARGETS nnkit_HDF5_export_action)
list(APPEND REQUIRED_TARGETS nnkit_HDF5_import_action)
list(APPEND REQUIRED_TARGETS i5diff)
TargetRequire_Return(${REQUIRED_TARGETS})

message(STATUS "Build onnx2tflite-integration-test: ON")

set(TEST_REPO "${ONNXTests_DIR}")
set(TEST_PBTXT_FILENAME "test.pbtxt")

unset(TESTCASES)

macro(add NAME)
  list(APPEND TESTCASES ${NAME})
endmacro(add)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(DEPS)
unset(KEYS)

foreach(PREFIX IN ITEMS ${TESTCASES})
  if(NOT IS_DIRECTORY "${TEST_REPO}/${PREFIX}")
    message(FATAL_ERROR "Missing '${PREFIX}' test")
  endif()

  set(PBTXT_SOURCE_PATH "${TEST_REPO}/${PREFIX}/${TEST_PBTXT_FILENAME}")

  set(PBTXT_FILE "${PREFIX}.pbtxt")
  set(PBTXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PBTXT_FILE}")

  set(ONNX_FILE "${PREFIX}.onnx")
  set(ONNX_PATH "${CMAKE_CURRENT_BINARY_DIR}/${ONNX_FILE}")

  # Copy .pbtxt
  add_custom_command(
    OUTPUT ${PBTXT_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${PBTXT_SOURCE_PATH}" "${PBTXT_PATH}"
    DEPENDS ${PBTXT_SOURCE_PATH}
    COMMENT "Generate ${PBTXT_FILE}")

  # Generate .onnx from .pbtxt
  add_custom_command(
    OUTPUT ${ONNX_PATH}
    COMMAND $<TARGET_FILE:onnxkit> encode ${PBTXT_PATH} ${ONNX_PATH}
    DEPENDS ${PBTXT_PATH}
    COMMENT "Generate ${ONNX_FILE}")

  list(APPEND DEPS ${INFO_PATH} ${ONNX_PATH})
  list(APPEND KEYS ${PREFIX})
endforeach(PREFIX)

##
## Copy testall
##
set(TEST_RUNNER_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh")
set(TEST_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/run")

add_custom_command(
  OUTPUT ${TEST_RUNNER}
  COMMAND ${CMAKE_COMMAND} -E copy "${TEST_RUNNER_TEMPLATE}" "${TEST_RUNNER}"
  DEPENDS ${TEST_RUNNER_TEMPLATE}
  COMMENT "Generate test runner")

list(APPEND DEPS "${TEST_RUNNER}")

###
### Generate test.config
###
set(TOOLCHAIN_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/toolchain.config")

add_custom_command(
  OUTPUT ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'NNKIT_RUN_PATH=\"$<TARGET_FILE:nnkit-run>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'ONNX_BACKEND_PATH=\"$<TARGET_FILE:nnkit_onnx_backend>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TFLITE_BACKEND_PATH=\"$<TARGET_FILE:nnkit_tflite_backend>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RANDOMIZE_ACTION_PATH=\"$<TARGET_FILE:nnkit_randomize_action>\"' >>
          ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_EXPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_export_action>\"' >>
          ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_IMPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_import_action>\"' >>
          ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'ONNX2TFLITE_PATH=\"$<TARGET_FILE:onnx2tflite>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'I5DIFF_PATH=\"$<TARGET_FILE:i5diff>\"' >> ${TOOLCHAIN_CONFIG}
  DEPENDS nnkit-run
          nnkit_onnx_backend
          nnkit_tflite_backend
          nnkit_randomize_action
          nnkit_HDF5_export_action
          nnkit_HDF5_import_action
          onnx2tflite
          i5diff
  COMMENT "Generate toolchain configuration")

list(APPEND DEPS "${TOOLCHAIN_CONFIG}")

# This "onnx2tflite_integration_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(onnx2tflite_integration_test_deps ALL DEPENDS ${DEPS})

# Run tests
add_test(NAME onnx2tflite_integration_test COMMAND "${TEST_RUNNER}" "${TOOLCHAIN_CONFIG}" ${KEYS})
