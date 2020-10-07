set(BCQ_TOOLS_FILES
    generate_bcq_metadata
    generate_bcq_output_arrays
)

foreach(BCQ_TOOLS IN ITEMS ${BCQ_TOOLS_FILES})

  set(BCQ_TOOLS_FILE ${BCQ_TOOLS})
  set(BCQ_TOOLS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${BCQ_TOOLS_FILE}")
  set(BCQ_TOOLS_BIN "${CMAKE_CURRENT_BINARY_DIR}/${BCQ_TOOLS_FILE}")
  set(BCQ_TOOLS_TARGET "${BCQ_TOOLS}_target")

  add_custom_command(OUTPUT ${BCQ_TOOLS_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${BCQ_TOOLS_SRC}" "${BCQ_TOOLS_BIN}"
    DEPENDS ${BCQ_TOOLS_SRC}
    COMMENT "Generate ${BCQ_TOOLS_BIN}"
  )

  add_custom_target(${BCQ_TOOLS_TARGET} ALL DEPENDS ${BCQ_TOOLS_BIN})

  install(FILES ${BCQ_TOOLS_BIN}
          PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
          DESTINATION bin)

endforeach(BCQ_TOOLS)
