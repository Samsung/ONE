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
    one-codegen
    one-infer
    one-prepare-venv
    onecc
)

foreach(ONE_COMMAND IN ITEMS ${ONE_COMMAND_FILES})

  set(ONE_COMMAND_FILE ${ONE_COMMAND})
  set(ONE_COMMAND_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ONE_COMMAND_FILE}")
  set(ONE_COMMAND_BIN "${CMAKE_CURRENT_BINARY_DIR}/${ONE_COMMAND_FILE}")
  set(ONE_COMMAND_TARGET "${ONE_COMMAND}_target")

  add_custom_command(OUTPUT ${ONE_COMMAND_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${ONE_COMMAND_SRC}" "${ONE_COMMAND_BIN}"
    DEPENDS ${ONE_COMMAND_SRC}
    COMMENT "Generate ${ONE_COMMAND_BIN}"
  )

  add_custom_target(${ONE_COMMAND_TARGET} ALL DEPENDS ${ONE_COMMAND_BIN})

  install(FILES ${ONE_COMMAND}
          PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
          DESTINATION bin)
          
endforeach(ONE_COMMAND)

set(ONE_UTILITY_FILES
    one-build.template.cfg
    onecc.template.cfg
    utils.py
    conv_mixin_1.8.0.patch
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
