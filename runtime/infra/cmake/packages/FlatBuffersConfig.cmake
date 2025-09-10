function(_FlatBuffers_import)
  if(TARGET flatbuffers::flatbuffers)
    message("FlatBuffers: already found")
    return()
  endif()

  if(NOT CMAKE_CROSSCOMPILING)
    find_package(Flatbuffers 23.5.26 QUIET)
    if(Flatbuffers_FOUND)
      set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
      return()
    endif(Flatbuffers_FOUND)
  endif(NOT CMAKE_CROSSCOMPILING)

  nnfw_find_package(FlatBuffersSource QUIET)
  if(NOT FlatBuffersSource_FOUND)
    message("FlatBuffers: cannot download Flatbuffers source")
    set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT FlatBuffersSource_FOUND)

  # From FlatBuffers's CMakeLists.txt
  set(FLATBUFFERS_BUILD_TESTS OFF)
  set(FLATBUFFERS_STATIC_FLATC ON)
  set(FLATBUFFERS_INSTALL OFF)
  if(CMAKE_CROSSCOMPILING)
    # Use host flatc on cross compiling
    set(FLATBUFFERS_BUILD_FLATC OFF)
  endif()

  add_subdirectory(${FlatBuffersSource_DIR} ${CMAKE_BINARY_DIR}/externals/flatbuffers)
  if(NOT TARGET flatbuffers)
    message("Flatbuffers: failed to build FlatBuffers")
    set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_CROSSCOMPILING)
    # Try to build flatc manually for host: set buildtool to gcc/g++ explicitly
    execute_process(
      COMMAND cmake -S ${FlatBuffersSource_DIR} -B ${CMAKE_BINARY_DIR}/externals/flatc-host
        -DFLATBUFFERS_BUILD_FLATC=ON -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_BUILD_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
      RESULT_VARIABLE FLATC_CONFIG_RESULT
    )
    if (NOT FLATC_CONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "Flatbuffers: failed to configure host flatc")
      set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
      return()
    endif()

    set(NUM_BUILD_THREADS 1)
    if(DEFINED EXTERNALS_BUILD_THREADS)
      set(NUM_BUILD_THREADS ${EXTERNALS_BUILD_THREADS})
    endif(DEFINED EXTERNALS_BUILD_THREADS)
    execute_process(
      COMMAND cmake --build ${CMAKE_BINARY_DIR}/externals/flatc-host -j ${NUM_BUILD_THREADS}
      RESULT_VARIABLE FLATC_BUILD_RESULT
    )
    if (NOT FLATC_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "Flatbuffers: failed to build host flatc")
      set(FlatBuffers_FOUND FALSE PARENT_SCOPE)
      return()
    endif()

    add_executable(flatbuffers::flatc IMPORTED GLOBAL)
    set_property(TARGET flatbuffers::flatc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(flatbuffers::flatc PROPERTIES
      IMPORTED_LOCATION_RELEASE "${CMAKE_BINARY_DIR}/externals/flatc-host/flatc"
    )
  endif()

  set_property(TARGET flatbuffers PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(flatbuffers PUBLIC $<$<CONFIG:Debug>:-Wno-sign-compare>)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)

  set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
  message("Flatbuffers: built FlatBuffers from source")
endfunction(_FlatBuffers_import)

_FlatBuffers_import()
