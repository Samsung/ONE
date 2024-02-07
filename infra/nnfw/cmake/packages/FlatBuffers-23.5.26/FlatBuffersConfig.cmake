function(_FlatBuffers_import)

  find_package(Flatbuffers QUIET)
  if(Flatbuffers_FOUND)
    set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
    return()
  endif(Flatbuffers_FOUND)

  nnas_find_package(FlatBuffersSource EXACT 23.5.26 QUIET)

  if(NOT FlatBuffersSource_FOUND)
    set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT FlatBuffersSource_FOUND)

  # From FlatBuffers's CMakeLists.txt
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/annotated_binary_text_gen.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/bfbs_gen_lua.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/bfbs_gen_nim.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/binary_annotator.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/code_generators.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/flatc.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_fbs.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/idl_gen_text.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/idl_parser.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/reflection.cpp")
  list(APPEND FlatBuffers_Library_SRCS "${FlatBuffersSource_DIR}/src/util.cpp")

  if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers ${FlatBuffers_Library_SRCS})
    target_include_directories(flatbuffers PUBLIC "${FlatBuffersSource_DIR}/include")
    set_property(TARGET flatbuffers PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_options(flatbuffers PUBLIC $<$<CONFIG:Debug>:-Wno-sign-compare>)

    add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
  endif(NOT TARGET flatbuffers::flatbuffers)

  set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffers_import)

_FlatBuffers_import()
