set(rawdata2hdf5_FILE "rawdata2hdf5")
set(rawdata2hdf5_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${rawdata2hdf5_FILE}")
set(rawdata2hdf5_BIN "${CMAKE_CURRENT_BINARY_DIR}/${rawdata2hdf5_FILE}")

add_custom_command(OUTPUT ${rawdata2hdf5_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${rawdata2hdf5_SRC}" "${rawdata2hdf5_BIN}"
    DEPENDS ${rawdata2hdf5_SRC}
    COMMENT "Generate ${rawdata2hdf5_BIN}"
  )

add_custom_target(rawdata2hdf5 ALL DEPENDS ${rawdata2hdf5_BIN})

install(FILES ${rawdata2hdf5_BIN}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION bin)
