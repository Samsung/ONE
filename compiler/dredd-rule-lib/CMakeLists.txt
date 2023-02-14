#
# copy rule-lib.sh (a library of shell script functions)
#
set(SOURCE_RULE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/rule-lib.sh")
set(TARGET_RULE_LIB "${CMAKE_CURRENT_BINARY_DIR}/rule-lib.sh")

add_custom_command(
  OUTPUT ${TARGET_RULE_LIB}
  COMMAND ${CMAKE_COMMAND} -E copy "${SOURCE_RULE_LIB}" "${TARGET_RULE_LIB}"
  DEPENDS ${SOURCE_RULE_LIB}
  COMMENT "Generate rule lib")

# Generate dependencies
add_custom_target(dredd_rule_lib ALL DEPENDS ${TARGET_RULE_LIB})

# How to get the path of rule-lib.sh in other CMakeLists.txt
#
# get_target_property(DREDD_RULE_LIB_DIR
#                     dredd_rule_lib BINARY_DIR)
# set(RULE_LIB_PATH "${DREDD_RULE_LIB_DIR}/rule-lib.sh")
