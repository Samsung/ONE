# TODO run when test is enabled

# DEPEND FILES
set(FILE_DEPS_VALCHK )

macro(COPY_SCRIPT FILENAME)
  set(SCRIPT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
  set(SCRIPT_DST "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
  add_custom_command(
    OUTPUT ${SCRIPT_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${SCRIPT_SRC}" "${SCRIPT_DST}"
    DEPENDS ${SCRIPT_SRC}
    COMMENT "tools/onnx2circle: prepare ${FILENAME}"
  )
  list(APPEND FILE_DEPS_VALCHK "${SCRIPT_DST}")
endmacro(COPY_SCRIPT)

# copy test scripts
COPY_SCRIPT(run_value_test.sh)
COPY_SCRIPT(exec_onnx.py)
COPY_SCRIPT(exec_circle.py)
COPY_SCRIPT(make_circle_input.py)
COPY_SCRIPT(util_h5_file.py)

add_custom_target(
  onnx2circle_value_test_target ALL
  DEPENDS onnx2circle_models_target ${FILE_DEPS_VALCHK}
)
