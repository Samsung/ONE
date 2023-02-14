nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tfkit)
list(APPEND REQUIRED_TARGETS tf2circle)
TargetRequire_Return(${REQUIRED_TARGETS})

nncc_find_resource(TensorFlowTests)

message(STATUS "Build tf2circle-ui-check: TRUE")
unset(TESTCASES)

macro(Add NAME)
  list(APPEND TESTCASES "${NAME}")
endmacro(Add)

include("test.lst" OPTIONAL)

##
## Generate configuration
##
set(CONFIG_PATH "${CMAKE_CURRENT_BINARY_DIR}/check.config")

add_custom_command(
  OUTPUT ${CONFIG_PATH}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CONFIG_PATH}
  COMMAND ${CMAKE_COMMAND} -E echo 'TESTCASE_BASE="${TensorFlowTests_DIR}"' >> ${CONFIG_PATH}
  COMMAND ${CMAKE_COMMAND} -E echo 'TFKIT_PATH=\"$<TARGET_FILE:tfkit>\"' >> ${CONFIG_PATH}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2CIRCLE_PATH=\"$<TARGET_FILE:tf2circle>\"' >> ${CONFIG_PATH}
  DEPENDS tfkit tf2circle
  COMMENT "Generate configuration")

## NOTE tf2circle_ui_check is not built by default
add_custom_target(
  tf2circle_ui_check
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/checkall.sh" "${CONFIG_PATH}" ${TESTCASES}
  DEPENDS "${CONFIG_PATH}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
