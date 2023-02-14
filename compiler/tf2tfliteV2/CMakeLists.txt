set(tf2tfliteV2_FILE "tf2tfliteV2.py")
set(tf2tfliteV2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${tf2tfliteV2_FILE}")
set(tf2tfliteV2_BIN "${CMAKE_CURRENT_BINARY_DIR}/${tf2tfliteV2_FILE}")

add_custom_command(
  OUTPUT ${tf2tfliteV2_BIN}
  COMMAND ${CMAKE_COMMAND} -E copy "${tf2tfliteV2_SRC}" "${tf2tfliteV2_BIN}"
  DEPENDS ${tf2tfliteV2_SRC}
  COMMENT "Generate ${tf2tfliteV2_BIN}")

add_custom_target(tf2tfliteV2 ALL DEPENDS ${tf2tfliteV2_BIN})

install(FILES ${tf2tfliteV2_BIN} DESTINATION bin)
