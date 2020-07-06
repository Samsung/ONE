set(tf2circleV2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tf2circleV2.py")
set(tf2circleV2_BIN "${CMAKE_CURRENT_BINARY_DIR}/tf2circleV2")

add_custom_command(OUTPUT ${tf2circleV2_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${tf2circleV2_SRC}" "${tf2circleV2_BIN}"
    DEPENDS ${tf2circleV2_SRC}
    COMMENT "Generate ${tf2circleV2_BIN}"
  )

add_custom_target(tf2circleV2 ALL DEPENDS ${tf2circleV2_BIN})

install(FILES ${tf2circleV2_BIN}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_WRITE GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION bin)
