if(NOT CIRCLE_MLIR_WORKDIR)
  if(NOT CIRCLE_MLIR_LOCALINST)
    set(FLATBUFFERS_INS "${EXTERNALS_BIN_DIR}/flatbuffers-install")
  else()
    set(FLATBUFFERS_INS "${CIRCLE_MLIR_LOCALINST}/flatbuffers-install")
  endif()
else()
  set(FLATBUFFERS_INS "${CIRCLE_MLIR_WORKDIR}")
endif()

set(FLATC_PATH "${FLATBUFFERS_INS}/bin/flatc")

link_directories(${FLATBUFFERS_INS}/lib)

add_library(flatbuffers INTERFACE)
target_include_directories(flatbuffers INTERFACE ${FLATBUFFERS_INS}/include)

function(FlatBuffers_Target TGT)
  set(oneValueArgs OUTPUT_DIR SCHEMA_DIR INCLUDE_DIR)
  set(multiValueArgs SCHEMA_FILES)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Use OUTPUT_DIR as INCLUDE_DIR if INCLUDE_DIR is not specified
  if(NOT ARG_INCLUDE_DIR)
    set(ARG_INCLUDE_DIR ${ARG_OUTPUT_DIR})
  endif(NOT ARG_INCLUDE_DIR)

  get_filename_component(abs_output_dir ${ARG_OUTPUT_DIR} ABSOLUTE)
  get_filename_component(abs_include_dir ${ARG_INCLUDE_DIR} ABSOLUTE)
  get_filename_component(abs_schema_dir ${ARG_SCHEMA_DIR} ABSOLUTE)

  # Let's reset list variables before using them
  # NOTE THIS DOES NOT AFFECT parent scope
  unset(SCHEMA_FILES)
  unset(OUTPUT_FILES)

  foreach(schema ${ARG_SCHEMA_FILES})
    get_filename_component(schema_fn "${schema}" NAME)
    get_filename_component(dir "${schema}" DIRECTORY)

    get_filename_component(schema_fn_we "${schema_fn}" NAME_WE)

    list(APPEND SCHEMA_FILES "${abs_schema_dir}/${schema}")
    list(APPEND OUTPUT_FILES "${abs_output_dir}/${schema_fn_we}_generated.h")
  endforeach()

  # Generate headers
  add_custom_command(OUTPUT ${OUTPUT_FILES}
                      COMMAND ${CMAKE_COMMAND} -E make_directory "${abs_output_dir}"
                      COMMAND "${FLATC_PATH}" -c --no-includes
                              --no-union-value-namespacing
                              --gen-mutable
                              --gen-object-api -o "${abs_output_dir}"
                              ${SCHEMA_FILES}
                      DEPENDS ${SCHEMA_FILES}
                      COMMENT "Generate '${TGT}' headers")

  # NOTE This header-only library is deliberately declared as STATIC library
  #      to avoid possible scope issues related with generated files
  add_library(${TGT} STATIC ${OUTPUT_FILES})
  set_target_properties(${TGT} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${TGT} PUBLIC "${ARG_INCLUDE_DIR}")
  target_link_libraries(${TGT} PUBLIC flatbuffers)
endfunction(FlatBuffers_Target)
