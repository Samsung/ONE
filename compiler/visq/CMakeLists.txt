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
    DEPENDS ${VISQ_PYTHON_FILE_SRC}
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
