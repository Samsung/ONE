set(imgdata2hdf5_FILE "imgdata2hdf5.py")
set(imgdata2hdf5_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${imgdata2hdf5_FILE}")
set(imgdata2hdf5_BIN "${CMAKE_CURRENT_BINARY_DIR}/${imgdata2hdf5_FILE}")

add_custom_command(OUTPUT ${imgdata2hdf5_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy "${imgdata2hdf5_SRC}" "${imgdata2hdf5_BIN}"
    DEPENDS ${imgdata2hdf5_SRC}
    COMMENT "Generate ${imgdata2hdf5_BIN}"
  )

add_custom_target(imgdata2hdf5 ALL DEPENDS ${imgdata2hdf5_BIN})

install(FILES ${imgdata2hdf5_BIN} DESTINATION bin)
