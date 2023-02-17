# NOTE these files should not have extensions.
#      below code will remove extension when copy and install.
set(ONE_COMMAND_FILES
    one-build
    one-import
    one-import-bcq
    one-import-tf
    one-import-tflite
    one-import-onnx
    one-optimize
    one-quantize
    one-pack
    one-partition
    one-profile
    one-infer
    one-codegen
    onecc
)

# TODO find better way for per-platform files
if(ONE_UBUNTU_CODENAME_JAMMY)
  # NOTE copy one-prepare-venv.u2204 as build/../one-prepare-venv
  #      and install build/../one-prepare-venv file
  list(APPEND ONE_COMMAND_FILES one-prepare-venv.u2204)
else()
  list(APPEND ONE_COMMAND_FILES one-prepare-venv)
endif()

# pytorch importer is an experimental feature, it is not used in default configuration
if(ENABLE_ONE_IMPORT_PYTORCH)
  list(APPEND ONE_COMMAND_FILES one-import-pytorch)
endif(ENABLE_ONE_IMPORT_PYTORCH)

foreach(ONE_COMMAND IN ITEMS ${ONE_COMMAND_FILES})

  set(ONE_COMMAND_FILE ${ONE_COMMAND})
  set(ONE_COMMAND_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ONE_COMMAND_FILE}")
  # strip extension from the name
  get_filename_component(ONE_COMMNAD_FILE_NAME ${ONE_COMMAND} NAME_WE)
  set(ONE_COMMAND_BIN "${CMAKE_CURRENT_BINARY_DIR}/${ONE_COMMNAD_FILE_NAME}")
  set(ONE_COMMAND_TARGET "${ONE_COMMAND}_target")

  add_custom_command(OUTPUT ${ONE_COMMAND_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${ONE_COMMAND_SRC}" "${ONE_COMMAND_BIN}"
    DEPENDS ${ONE_COMMAND_SRC}
    COMMENT "Generate ${ONE_COMMAND_BIN}"
  )

  add_custom_target(${ONE_COMMAND_TARGET} ALL DEPENDS ${ONE_COMMAND_BIN})

  install(FILES ${ONE_COMMAND_BIN}
          PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
          DESTINATION bin)
          
endforeach(ONE_COMMAND)

set(ONE_UTILITY_FILES
    one-build.template.cfg
    onecc.template.cfg
    onnx_legalizer.py
)

foreach(ONE_UTILITY IN ITEMS ${ONE_UTILITY_FILES})

  set(ONE_UTILITY_FILE ${ONE_UTILITY})
  set(ONE_UTILITY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ONE_UTILITY_FILE}")
  set(ONE_UTILITY_BIN "${CMAKE_CURRENT_BINARY_DIR}/${ONE_UTILITY_FILE}")
  set(ONE_UTILITY_TARGET "${ONE_UTILITY}_target")

  add_custom_command(OUTPUT ${ONE_UTILITY_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${ONE_UTILITY_SRC}" "${ONE_UTILITY_BIN}"
    DEPENDS ${ONE_UTILITY_SRC}
    COMMENT "Generate ${ONE_UTILITY_BIN}"
  )

  add_custom_target(${ONE_UTILITY_TARGET} ALL DEPENDS ${ONE_UTILITY_BIN})

  install(FILES ${ONE_UTILITY}
          PERMISSIONS OWNER_WRITE OWNER_READ
                      GROUP_READ
                      WORLD_READ
          DESTINATION bin)

endforeach(ONE_UTILITY)

# one-pack internally uses model2nnpkg tool
set(MODEL2NNPKG "${NNAS_PROJECT_SOURCE_DIR}/tools/nnpackage_tool/model2nnpkg/model2nnpkg.py")
install(FILES ${MODEL2NNPKG}
              PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE
              DESTINATION bin
              RENAME "model2nnpkg")

# make python directory
set(ONE_PYTHON_FILES constant.py
                     make_cmd.py
                     CfgRunner.py
                     OptionBuilder.py
                     TopologicalSortHelper.py
                     WorkflowRunner.py
                     utils.py
                     O1.txt)

foreach(ONE_PYTHON_FILE IN ITEMS ${ONE_PYTHON_FILES})

  set(ONE_PYTHON_DIR "onelib")
  set(ONE_PYTHON_DIR_BIN "${CMAKE_CURRENT_BINARY_DIR}/${ONE_PYTHON_DIR}")
  set(ONE_PYTHON_FILE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ONE_PYTHON_DIR}/${ONE_PYTHON_FILE}")
  set(ONE_PYTHON_FILE_BIN "${CMAKE_CURRENT_BINARY_DIR}/${ONE_PYTHON_DIR}/${ONE_PYTHON_FILE}")
  set(ONE_PYTHON_TARGET "${ONE_PYTHON_FILE}_target")

  add_custom_command(OUTPUT ${ONE_PYTHON_DIR_BIN}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ONE_PYTHON_DIR_BIN}"
    COMMENT "Generate ${ONE_PYTHON_DIR_BIN}"
  )

  add_custom_command(OUTPUT ${ONE_PYTHON_FILE_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${ONE_PYTHON_FILE_SRC}" "${ONE_PYTHON_FILE_BIN}"
    DEPENDS ${ONE_PYTHON_SRC}
    COMMENT "Generate ${ONE_PYTHON_FILE_BIN}"
  )

  add_custom_target(${ONE_PYTHON_TARGET} ALL DEPENDS ${ONE_PYTHON_DIR_BIN} ${ONE_PYTHON_FILE_BIN})

  install(DIRECTORY ${ONE_PYTHON_DIR}
          FILE_PERMISSIONS OWNER_WRITE OWNER_READ
                           GROUP_READ
                           WORLD_READ
          DESTINATION bin)

endforeach(ONE_PYTHON_FILE)

set(ONE_DOCUMENT_FILES
    how-to-use-one-commands.txt
    how-to-prepare-virtualenv.txt
    how-to-create-hdf5-dataset.txt
)

foreach(ONE_DOCUMENT IN ITEMS ${ONE_DOCUMENT_FILES})

  install(FILES ${ONE_DOCUMENT} DESTINATION doc)

endforeach(ONE_DOCUMENT)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_subdirectory(dummy-driver)
add_subdirectory(tests)
add_subdirectory(validate-onnx2circle)

# Copy O1.txt file to test directory
set(ONELIB_O1_FILE ${CMAKE_CURRENT_BINARY_DIR}/onelib/O1.txt)
set(TEST_O1_FILE ${CMAKE_CURRENT_BINARY_DIR}/tests/O1.txt)

add_custom_command(OUTPUT ${TEST_O1_FILE}
  COMMAND ${CMAKE_COMMAND} -E copy ${ONELIB_O1_FILE} ${TEST_O1_FILE}
  DEPENDS ${ONELIB_O1_FILE}
  COMMENT "Copy O1.txt to tests directory"
)

add_custom_target(TGT_TEST_O1_FILE ALL DEPENDS ${TEST_O1_FILE})

install(FILES ${TEST_O1_FILE}
        PERMISSIONS OWNER_READ
        GROUP_READ
        WORLD_READ
        DESTINATION test)
