function(_FlatBuffers_import)
  if(TARGET flatbuffers::flatbuffers)
    # Already found
    return()
  endif()

  find_package(Flatbuffers 23.5.26 QUIET)
  if(Flatbuffers_FOUND)
    message(STATUS "Flatbuffers: found Flatbuffers")
    set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
    return()
  endif(Flatbuffers_FOUND)

  nnfw_find_package(FlatBuffersSource QUIET)

  if(NOT FlatBuffersSource_FOUND)
    message(STATUS "Flatbuffers: cannot find FlatBuffers source")
    set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT FlatBuffersSource_FOUND)

  set(FLATBUFFERS_BUILD_TESTS OFF)
  set(FLATBUFFERS_STATIC_FLATC ON)
  set(FLATBUFFERS_INSTALL OFF)
  set(FLATBUFFERS_BUILD_FLATC OFF)
  add_subdirectory(${FlatBuffersSource_DIR} ${CMAKE_BINARY_DIR}/externals/flatbuffers)
  if(NOT TARGET flatbuffers)
    message(STATUS "Flatbuffers: failed to build FlatBuffers")
    set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  set_property(TARGET flatbuffers PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(flatbuffers PUBLIC $<$<CONFIG:Debug>:-Wno-sign-compare>)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)

  message(STATUS "Flatbuffers: built FlatBuffers from source")
  set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffers_import)

_FlatBuffers_import()
