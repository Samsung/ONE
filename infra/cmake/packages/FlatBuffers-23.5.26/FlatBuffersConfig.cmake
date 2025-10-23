# TODO Remove other Flatbuffers versions
function(_FlatBuffers_import)
  if(TARGET flatbuffers-23.5.26)
    # Already found
    set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
    return()
  endif()

  set(FlatBuffers_FOUND FALSE PARENT_SCOPE)

  # Don't use pre-installed FlatBuffers when cross-compiling
  if(NOT CMAKE_CROSSCOMPILING)
    # Clear to avoid infinite recursion
    # Not need to backup & restore cache value
    # - We will use same flatbuffers/flatc setting here with installed package on native build
    # - If we fail to find installed package, cache value will be filled again on 2nd attempt,
    #   and will not reach here again because of above TARGET checking condition
    unset(FlatBuffers_DIR CACHE)
    find_package(FlatBuffers EXACT 23.5.26 QUIET NO_CMAKE_PATH)
    if(FlatBuffers_FOUND)
      message(STATUS "Found FlatBuffers ${FlatBuffers_FIND_VERSION}")
      add_library(flatbuffers-23.5.26 ALIAS flatbuffers::flatbuffers)
      add_executable(flatc-23.5.26 ALIAS flatbuffers::flatc)
      set(FlatBuffers_FOUND TRUE PARENT_SCOPE)
      return()
    endif(FlatBuffers_FOUND)
  endif(NOT CMAKE_CROSSCOMPILING)
endfunction(_FlatBuffers_import)

function(_FlatBuffers_build)
  if(NOT BUILD_FLATBUFFERS)
    message(STATUS "FlatBuffersConfig !BUILD_FLATBUFFERS")
    return()
  endif(NOT BUILD_FLATBUFFERS)

  if(TARGET flatbuffers-23.5.26)
    # Already built
    return()
  endif()

  nnas_find_package(FlatBuffersSource EXACT 23.5.26 QUIET)

  if(NOT FlatBuffersSource_FOUND)
    # Source is not available
    message(STATUS "FlatBuffersConfig !FlatBuffersSource_FOUND")
    return()
  endif(NOT FlatBuffersSource_FOUND)

  set(FLATBUFFERS_BUILD_TESTS OFF)
  set(FLATBUFFERS_STATIC_FLATC ON)
  set(FLATBUFFERS_INSTALL OFF)
  if(CMAKE_CROSSCOMPILING)
    set(FLATBUFFERS_BUILD_FLATC OFF)
  endif(CMAKE_CROSSCOMPILING)
  add_subdirectory(${FlatBuffersSource_DIR} ${CMAKE_BINARY_DIR}/externals/FLATBUFFERS-23.5.26/build)
  if(NOT TARGET flatbuffers)
    message(STATUS "Flatbuffers: failed to build FlatBuffers")
    return()
  endif()

  set_property(TARGET flatbuffers PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(flatbuffers PUBLIC $<$<CONFIG:Debug>:-Wno-sign-compare>)
  add_library(flatbuffers-23.5.26 ALIAS flatbuffers)

  if(CMAKE_CROSSCOMPILING)
    # Build flatc for host manually: set buildtool to gcc/g++ explicitly
    message(STATUS "Flatbuffers: build flatbuffers for host...")
    # Use ${FlatBufferSource_VERSION} as suffix to distinguish version change
    set(FLATC_HOST_BINARY_DIR ${CMAKE_BINARY_DIR}/externals/FLATC-HOST-${FlatBuffersSource_VERSION})
    execute_process(
      COMMAND cmake -S ${FlatBuffersSource_DIR} -B ${FLATC_HOST_BINARY_DIR}
        -DFLATBUFFERS_BUILD_FLATC=ON -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_BUILD_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
      RESULT_VARIABLE FLATC_CONFIG_RESULT
    )
    if (NOT FLATC_CONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "Flatbuffers: failed to configure host flatc")
      return()
    endif()

    set(NUM_BUILD_THREADS 1)
    if(DEFINED EXTERNALS_BUILD_THREADS)
      set(NUM_BUILD_THREADS ${EXTERNALS_BUILD_THREADS})
    endif(DEFINED EXTERNALS_BUILD_THREADS)
    execute_process(
      COMMAND cmake --build ${FLATC_HOST_BINARY_DIR} -j ${NUM_BUILD_THREADS}
      RESULT_VARIABLE FLATC_BUILD_RESULT
    )
    if (NOT FLATC_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "Flatbuffers: failed to build host flatc")
      return()
    endif()

    add_executable(flatc-23.5.26 IMPORTED GLOBAL)
    set_property(TARGET flatc-23.5.26 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(flatc-23.5.26 PROPERTIES
      IMPORTED_LOCATION_RELEASE "${FLATC_HOST_BINARY_DIR}/flatc"
    )
  else()
    add_executable(flatc-23.5.26 ALIAS flatc)
  endif()

  message(STATUS "Flatbuffers: built FlatBuffers from source")
endfunction(_FlatBuffers_build)

_FlatBuffers_build()
_FlatBuffers_import()

set(FLATC_PATH "$<TARGET_FILE:flatc-23.5.26>")
if(FlatBuffers_FOUND)
  if(NOT TARGET flatbuffers-23.5.26)
    add_library(flatbuffers-23.5.26 INTERFACE)
    target_link_libraries(flatbuffers-23.5.26 INTERFACE flatbuffers::flatbuffers)
    message(STATUS "Found flatbuffers-23.5.26: TRUE")
  endif(NOT TARGET flatbuffers-23.5.26)

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
                       COMMAND "${FLATC_PATH}" -c --no-includes
                       --no-union-value-namespacing
                       --gen-object-api -o "${abs_output_dir}"
                       ${SCHEMA_FILES}
                       DEPENDS flatbuffers::flatc)

    set(${PREFIX}_SOURCES ${OUTPUT_FILES} PARENT_SCOPE)
    set(${PREFIX}_INCLUDE_DIRS ${abs_output_dir} PARENT_SCOPE)
  endfunction(FlatBuffers_Generate)

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
                               --gen-object-api -o "${abs_output_dir}"
                               ${SCHEMA_FILES}
                       DEPENDS ${SCHEMA_FILES}
                       COMMENT "Generate '${TGT}' headers")

    # NOTE This header-only library is deliberately declared as STATIC library
    #      to avoid possible scope issues related with generated files
    add_library(${TGT} STATIC ${OUTPUT_FILES})
    set_target_properties(${TGT} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(${TGT} PUBLIC "${ARG_INCLUDE_DIR}")
    target_link_libraries(${TGT} PUBLIC flatbuffers-23.5.26)
  endfunction(FlatBuffers_Target)

  function(FlatBuffersMuteable_Target TGT)
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
                               --gen-object-api
                               --gen-mutable
                               -o "${abs_output_dir}"
                               ${SCHEMA_FILES}
                       DEPENDS ${SCHEMA_FILES}
                       COMMENT "Generate '${TGT}' headers")

    # NOTE This header-only library is deliberately declared as STATIC library
    #      to avoid possible scope issues related with generated files
    add_library(${TGT} STATIC ${OUTPUT_FILES})
    set_target_properties(${TGT} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(${TGT} PUBLIC "${ARG_INCLUDE_DIR}")
    target_link_libraries(${TGT} PUBLIC flatbuffers-23.5.26)
  endfunction(FlatBuffersMuteable_Target)
endif(FlatBuffers_FOUND)
