function(_FlatBuffers_import)

  find_package(Flatbuffers QUIET)

  if(Flatbuffers_FOUND)
    set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
    return()
  endif(Flatbuffers_FOUND)

  # NOTE Tizen uses 1.11
  nnas_find_package(FlatBuffersSource EXACT 1.11 QUIET)

  if(NOT FlatBuffersSource_FOUND)
    set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT FlatBuffersSource_FOUND)

  # From FlatBuffers's CMakeLists.txt
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/code_generators.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/idl_parser.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_text.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/reflection.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/util.cpp")

  # From FlatBuffers's CMakeLists.txt
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_cpp.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_dart.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_general.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_go.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_js_ts.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_php.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_python.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_lobster.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_lua.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_rust.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_fbs.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_grpc.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_json_schema.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/flatc.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/src/flatc_main.cpp")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/grpc/src/compiler/cpp_generator.cc")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/grpc/src/compiler/go_generator.cc")
  list(APPEND FlatBuffers_Compiler_SRCS "${FlatBuffersSource_DIR}/grpc/src/compiler/java_generator.cc")

  if(NOT TARGET flatbuffers)
    add_library(flatbuffers ${FlatBuffers_Library_SRCS})
    target_include_directories(flatbuffers PUBLIC "${FlatBuffersSource_DIR}/include")
    set_property(TARGET flatbuffers PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif(NOT TARGET flatbuffers)

  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)

  if(NOT TARGET flatc)
    add_executable(flatc ${FlatBuffers_Compiler_SRCS})
    target_include_directories(flatc PRIVATE "${FlatBuffersSource_DIR}/grpc")
    target_link_libraries(flatc flatbuffers::flatbuffers)
  endif(NOT TARGET flatc)

  add_executable(flatbuffers::flatc ALIAS flatc)

  set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffers_import)

_FlatBuffers_import()

if(FlatBuffers_FOUND)
  function(FlatBuffers_Generate PREFIX OUTPUT_DIR SCHEMA_DIR)
    get_filename_component(abs_output_dir ${OUTPUT_DIR} ABSOLUTE)
    get_filename_component(abs_schema_dir ${SCHEMA_DIR} ABSOLUTE)

    foreach(schema ${ARGN})
      get_filename_component(schema_fn "${schema}" NAME)
      get_filename_component(dir "${schema}" DIRECTORY)

      get_filename_component(schema_fn_we "${schema_fn}" NAME_WE)

      list(APPEND SCHEMA_FILES "${abs_schema_dir}/${schema}")
      list(APPEND OUTPUT_FILES "${abs_output_dir}/${schema_fn_we}_generated.h")
    endforeach()

    add_custom_command(OUTPUT ${OUTPUT_FILES}
                       COMMAND ${CMAKE_COMMAND} -E make_directory "${abs_output_dir}"
                       COMMAND "$<TARGET_FILE:flatc>" -c --no-includes
                       --no-union-value-namespacing
                       --gen-object-api -o "${abs_output_dir}"
                       ${SCHEMA_FILES}
                       DEPENDS flatbuffers::flatc)

    set(${PREFIX}_SOURCES ${OUTPUT_FILES} PARENT_SCOPE)
    set(${PREFIX}_INCLUDE_DIRS ${abs_output_dir} PARENT_SCOPE)
  endfunction(FlatBuffers_Generate)
endif(FlatBuffers_FOUND)
