# TODO Remove this line
return()

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(VISQ_TEST_DEPS)

###
### Copy test files
###
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_BINARY_DIR}/test)

list(APPEND VISQ_TEST_DEPS ${CMAKE_CURRENT_BINARY_DIR}/test)

###
### Import visqlib module
###
get_target_property(VISQ_BIN_PATH visq BINARY_DIR)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/visqlib
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${VISQ_BIN_PATH}/visqlib ${CMAKE_CURRENT_BINARY_DIR}/visqlib)

list(APPEND VISQ_TEST_DEPS ${CMAKE_CURRENT_BINARY_DIR}/visqlib)

###
### Import pics module
###
get_target_property(PICS_BIN_PATH pics BINARY_DIR)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/circle
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${PICS_BIN_PATH}/circle ${CMAKE_CURRENT_BINARY_DIR}/circle)

list(APPEND VISQ_TEST_DEPS ${CMAKE_CURRENT_BINARY_DIR}/circle)

###
### Generate Resources.py
###
set(RESOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/test/Resources.py")

get_target_property(FP32_MODEL_DIR testDataGenerator BINARY_DIR)

add_custom_command(
  OUTPUT ${RESOURCE_FILE}
  COMMAND ${CMAKE_COMMAND} -E echo 'fp32_model_dir=\"${FP32_MODEL_DIR}\"' >> ${RESOURCE_FILE}
  COMMENT "Generate file to specify resource location"
)

list(APPEND VISQ_TEST_DEPS ${RESOURCE_FILE})

add_custom_target(visq_unittest ALL DEPENDS ${VISQ_TEST_DEPS})

# Use Python in venv to run unittest with pydot module
add_test(
  NAME visq_unittest
  COMMAND ${NNCC_OVERLAY_DIR}/venv_2_8_0/bin/python -m unittest
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

if(ONE_UBUNTU_CODENAME_JAMMY)
  add_test(
    NAME visq_210_unittest
    COMMAND ${NNCC_OVERLAY_DIR}/venv_2_10_1/bin/python -m unittest
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif(ONE_UBUNTU_CODENAME_JAMMY)
