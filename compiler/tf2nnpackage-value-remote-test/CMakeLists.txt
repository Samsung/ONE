include("test.lst")

# Do not make test if there are no remote machine information
if(NOT REMOTE_IP)
  message(STATUS "tf2nnpackage-value-remote-test: need to set REMOTE IP")
  return()
endif(NOT REMOTE_IP)

if(NOT REMOTE_USER)
  message(STATUS "tf2nnpackage-value-remote-test: need to set REMOTE_USER")
  return()
endif(NOT REMOTE_USER)

nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS testDataGenerator)
TargetRequire_Return(${REQUIRED_TARGETS})

message(STATUS "tf2nnpackage-value-remote-test: run tests")

unset(TEST_NAMES)

nncc_find_resource(TensorFlowLiteRecipes)
set(TFLITE_RECIPE_REPO "${TensorFlowLiteRecipes_DIR}")

file(GLOB SUBDIR RELATIVE ${TFLITE_RECIPE_REPO} ${TFLITE_RECIPE_REPO}/*)
foreach(DIR IN ITEMS ${SUBDIR})
  if(IS_DIRECTORY ${TFLITE_RECIPE_REPO}/${DIR})
    list(APPEND TEST_NAMES ${DIR})
  endif()
endforeach()

get_target_property(ARTIFACTS_SRC_PATH testDataGenerator SOURCE_DIR)

# In this test, only the runtime test is performed because the test from tf to
# nnpackage is done in common-artifacts, and for this runtime test, generation of
# test data is required. And, tcgenerate in ${ARTIFACTS_SRC_PATH}/exclude.lst
# means it won't generate test data, which is why below "tcgenerate" macro excludes
# specific opearators from runtime test.
# Also, since circlize and optimize macro included in `exclude.lst` file is only
# needed in common-artifacts, it has no function here.
macro(circlize)
endmacro()
macro(optimize)
endmacro()

macro(tcgenerate NAME)
  list(REMOVE_ITEM TEST_NAMES ${NAME})
endmacro()

include("${ARTIFACTS_SRC_PATH}/exclude.lst")

# Copy testall
set(TEST_RUNNER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh")
set(TEST_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/testall.sh")

add_custom_command(
  OUTPUT ${TEST_RUNNER}
  COMMAND ${CMAKE_COMMAND} -E copy "${TEST_RUNNER_SOURCE}" "${TEST_RUNNER}"
  DEPENDS ${TEST_RUNNER_SOURCE}
  COMMENT "Generate test runner"
)

list(APPEND TEST_DEPS "${TEST_RUNNER}")

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

# Generate test.config
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RUNTIME_LIBRARY_PATH=\"${NNAS_PROJECT_SOURCE_DIR}/Product/out/\"' >> ${TEST_CONFIG}
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tf2nnpackage_value_remote_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tf2nnpackage_value_remote_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME tf2nnpackage_value_remote_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          "${REMOTE_IP}"
          "${REMOTE_USER}"
          ${TEST_NAMES}
)
