function(_FlatBuffers_import)

  find_package(Flatbuffers QUIET)
  if(Flatbuffers_FOUND)
    set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
    return()
  endif(Flatbuffers_FOUND)

  # NOTE Tizen uses 2.0
  nnas_find_package(FlatBuffersSource EXACT 2.0 QUIET)

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

  if(NOT TARGET flatbuffers::flatbuffers-2.0)
    add_library(flatbuffers-2.0 ${FlatBuffers_Library_SRCS})
    target_include_directories(flatbuffers-2.0 PUBLIC "${FlatBuffersSource_DIR}/include")
    set_property(TARGET flatbuffers PROPERTY POSITION_INDEPENDENT_CODE ON)

    add_library(flatbuffers::flatbuffers-2.0 ALIAS flatbuffers-2.0)
  endif(NOT TARGET flatbuffers::flatbuffers-2.0)

  set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffers_import)

_FlatBuffers_import()
