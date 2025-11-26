set(EXT_FILES
    one-import-onnx-ext
)

# NOTE using loop is to prepare
foreach(EXT_FILE IN ITEMS ${EXT_FILES})
  set(EXT_FILE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${EXT_FILE}")

  # strip extension from the name if exist
  get_filename_component(EXT_FILE_NAME ${EXT_FILE} NAME_WE)
  set(EXT_FILE_BIN "${CMAKE_CURRENT_BINARY_DIR}/${EXT_FILE_NAME}")

  add_custom_command(OUTPUT ${EXT_FILE_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${EXT_FILE_SRC}" "${EXT_FILE_BIN}"
    DEPENDS ${EXT_FILE_SRC}
    COMMENT "Generate ${EXT_FILE_BIN}"
  )

  set(EXT_FILE_TARGET "${EXT_FILE}_target")
  add_custom_target(${EXT_FILE_TARGET} ALL DEPENDS ${EXT_FILE_BIN})

  install(FILES ${EXT_FILE_BIN}
          PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
          DESTINATION bin)

endforeach()
