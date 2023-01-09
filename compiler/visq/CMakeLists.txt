unset(VISQ_DEPS)

###
### Set up visq executable
###
set(VISQ_FILE "visq")
set(VISQ_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${VISQ_FILE}")
set(VISQ_BIN "${CMAKE_CURRENT_BINARY_DIR}/${VISQ_FILE}")

add_custom_command(OUTPUT ${VISQ_BIN}
  COMMAND ${CMAKE_COMMAND} -E copy "${VISQ_SRC}" "${VISQ_BIN}"
  DEPENDS ${VISQ_SRC}
  COMMENT "Generate ${VISQ_BIN}"
)

list(APPEND VISQ_DEPS ${VISQ_BIN})

###
### Set up visqlib directory
###
set(VISQ_PYTHON_DIR "visqlib")
set(VISQ_PYTHON_DIR_BIN "${CMAKE_CURRENT_BINARY_DIR}/${VISQ_PYTHON_DIR}")

add_custom_command(OUTPUT ${VISQ_PYTHON_DIR_BIN}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VISQ_PYTHON_DIR_BIN}"
  COMMENT "Generate ${VISQ_PYTHON_DIR_BIN}"
)

list(APPEND VISQ_DEPS ${VISQ_PYTHON_DIR_BIN})

###
### Set up Python files
###
set(VISQ_PYTHON_FILES DumpFakeQuantFM.py
                      DumpFP32FM.py
                      Palette.py
                      QErrorComputer.py
                      DotBuilder.py
                      Util.py)

foreach(VISQ_PYTHON_FILE IN ITEMS ${VISQ_PYTHON_FILES})
  set(VISQ_PYTHON_FILE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${VISQ_PYTHON_DIR}/${VISQ_PYTHON_FILE}")
  set(VISQ_PYTHON_FILE_BIN "${CMAKE_CURRENT_BINARY_DIR}/${VISQ_PYTHON_DIR}/${VISQ_PYTHON_FILE}")

  add_custom_command(OUTPUT ${VISQ_PYTHON_FILE_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${VISQ_PYTHON_FILE_SRC}" "${VISQ_PYTHON_FILE_BIN}"
    DEPENDS ${VISQ_PYTHON_SRC}
    COMMENT "Generate ${VISQ_PYTHON_FILE_BIN}"
  )

  list(APPEND VISQ_DEPS ${VISQ_PYTHON_FILE_BIN})

endforeach(VISQ_PYTHON_FILE)

add_custom_target(visq ALL DEPENDS ${VISQ_DEPS})

install(FILES ${VISQ_FILE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION bin)

install(DIRECTORY ${VISQ_PYTHON_DIR}
        FILE_PERMISSIONS OWNER_WRITE OWNER_READ
                          GROUP_READ
                          WORLD_READ
        DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

find_package(PythonInterp 3.8 QUIET)

unset(VISQ_TEST_DEPS)

###
### Copy test files
###
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_BINARY_DIR}/test)

list(APPEND VISQ_TEST_DEPS ${CMAKE_CURRENT_BINARY_DIR}/test)

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

add_test(
  NAME visq_unittest
  COMMAND ${NNCC_OVERLAY_DIR}/venv_2_8_0/bin/python -m unittest
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
