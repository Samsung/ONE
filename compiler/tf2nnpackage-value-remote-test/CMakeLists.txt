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
  COMMAND ${CMAKE_COMMAND} -E echo 'NNPKG_TEST_PATH=\"${NNAS_PROJECT_SOURCE_DIR}/tests/scripts/nnpkg_test.sh\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RUNTIME_LIBRARY_PATH=\"${NNAS_PROJECT_SOURCE_DIR}/Product/out/\"' >> ${TEST_CONFIG}
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tf2nnpackage_value_remote_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tf2nnpackage_value_remote_test_deps ALL DEPENDS ${TEST_DEPS})

include("test.lst")

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
