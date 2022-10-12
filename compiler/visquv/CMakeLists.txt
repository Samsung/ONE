set(VISQUV_FILES
    visquv-server.js
    visquv-index.html
    visquv-index.js
    test_error.json
    test_error.circle
)

foreach(VISQUV_ITEM IN ITEMS ${VISQUV_FILES})

  set(VISQUV_FILE src/${VISQUV_ITEM})
  set(VISQUV_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${VISQUV_FILE}")
  set(VISQUV_BIN "${CMAKE_CURRENT_BINARY_DIR}/${VISQUV_FILE}")
  set(VISQUV_TARGET "${VISQUV_ITEM}_target")

  add_custom_command(OUTPUT ${VISQUV_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${VISQUV_SRC}" "${VISQUV_BIN}"
    DEPENDS ${VISQUV_SRC}
    COMMENT "Generate ${VISQUV_BIN}"
  )

  add_custom_target(${VISQUV_TARGET} ALL DEPENDS ${VISQUV_BIN})

  install(FILES ${VISQUV_FILE}
          PERMISSIONS OWNER_WRITE OWNER_READ
                      GROUP_READ
                      WORLD_READ
          DESTINATION bin/visquv)

endforeach(VISQUV_ITEM)

set(VISQUV_EXT_FILES
    dagre.js
    base.js
    text.js
    json.js
    xml.js
    python.js
    protobuf.js
    flatbuffers.js
    flexbuffers.js
    zip.js
    gzip.js
    tar.js
    view-grapher.js
    view-sidebar.js
    view.js
    view-grapher.css
    view-sidebar.css
    circle.js
    circle-metadata.json
    circle-schema.js
)

foreach(VISQUV_ITEM IN ITEMS ${VISQUV_EXT_FILES})

  set(VISQUV_FILE src/external/${VISQUV_ITEM})
  set(VISQUV_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${VISQUV_FILE}")
  set(VISQUV_BIN "${CMAKE_CURRENT_BINARY_DIR}/${VISQUV_FILE}")
  set(VISQUV_TARGET "${VISQUV_ITEM}_target")

  add_custom_command(OUTPUT ${VISQUV_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${VISQUV_SRC}" "${VISQUV_BIN}"
    DEPENDS ${VISQUV_SRC}
    COMMENT "Generate ${VISQUV_BIN}"
  )

  add_custom_target(${VISQUV_TARGET} ALL DEPENDS ${VISQUV_BIN})

  install(FILES ${VISQUV_FILE}
          PERMISSIONS OWNER_WRITE OWNER_READ
                      GROUP_READ
                      WORLD_READ
          DESTINATION bin/visquv)

endforeach(VISQUV_ITEM)

set(COMMAND_FILES
    one-visquv
)

foreach(COMMAND_ITEM IN ITEMS ${COMMAND_FILES})

  set(COMMAND_FILE ${COMMAND_ITEM})
  set(COMMAND_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${COMMAND_FILE}")
  set(COMMAND_BIN "${CMAKE_CURRENT_BINARY_DIR}/${COMMAND_FILE}")
  set(COMMAND_TARGET "${COMMAND_ITEM}_target")

  add_custom_command(OUTPUT ${COMMAND_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${COMMAND_SRC}" "${COMMAND_BIN}"
    DEPENDS ${COMMAND_SRC}
    COMMENT "Generate ${COMMAND_BIN}"
  )

  add_custom_target(${COMMAND_TARGET} ALL DEPENDS ${COMMAND_BIN})

  install(FILES ${COMMAND_FILE}
          PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
          DESTINATION bin)

endforeach(COMMAND_ITEM)
