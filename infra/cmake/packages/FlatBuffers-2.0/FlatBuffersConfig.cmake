function(_FlatBuffers_import)
  find_package(Flatbuffers 2.0 EXACT QUIET PATHS ${EXT_OVERLAY_DIR}/FLATBUFFERS-2.0)
  set(FlatBuffers_FOUND ${Flatbuffers_FOUND} PARENT_SCOPE)
endfunction(_FlatBuffers_import)

function(_FlatBuffers_build)
  if(NOT BUILD_FLATBUFFERS)
    message(STATUS "FlatBuffersConfig !BUILD_FLATBUFFERS")
    return()
  endif(NOT BUILD_FLATBUFFERS)

  nnas_find_package(FlatBuffersSource EXACT 2.0 QUIET)

  if(NOT FlatBuffersSource_FOUND)
    # Source is not available
    message(STATUS "FlatBuffersConfig !FlatBuffersSource_FOUND")
    return()
  endif(NOT FlatBuffersSource_FOUND)

  set(ADDITIONAL_CXX_FLAGS "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
    set(ADDITIONAL_CXX_FLAGS "-Wno-error=class-memaccess")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 12.0)
    set(ADDITIONAL_CXX_FLAGS "-Wno-error=stringop-overflow")
  endif()

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${FlatBuffersSource_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/FLATBUFFERS-2.0/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}/FLATBUFFERS-2.0
                      BUILD_FLAGS ${ADDITIONAL_CXX_FLAGS}
                      IDENTIFIER  "2.0"
                      EXTRA_OPTS  "-DFLATBUFFERS_BUILD_TESTS:BOOL=OFF"
                                  "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
                      PKG_NAME    "FLATBUFFERS-2.0")

endfunction(_FlatBuffers_build)

_FlatBuffers_build()
_FlatBuffers_import()

# for cross compilation BUILD_HOST_EXEC should be set for host flatc executable
# flatc should exist as ${BUILD_HOST_EXEC}/overlay/bin/flatc.
# and then if EXTERNAL_FLATC is set then use ${EXTERNAL_FLATC} file.
set(FLATC_PATH "$<TARGET_FILE:flatbuffers::flatc>")

if(DEFINED ENV{BUILD_HOST_EXEC})
  set(FLATC_PATH $ENV{BUILD_HOST_EXEC}/overlay/FLATBUFFERS-2.0/bin/flatc)
endif(DEFINED ENV{BUILD_HOST_EXEC})
if(DEFINED ENV{EXTERNAL_FLATC})
  set(FLATC_PATH $ENV{EXTERNAL_FLATC})
endif(DEFINED ENV{EXTERNAL_FLATC})

if(FlatBuffers_FOUND)
  if(NOT TARGET flatbuffers-2.0)
    add_library(flatbuffers-2.0 INTERFACE)
    target_link_libraries(flatbuffers-2.0 INTERFACE flatbuffers::flatbuffers)
    message(STATUS "Found flatbuffers-2.0: TRUE")
  endif(NOT TARGET flatbuffers-2.0)

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
    target_link_libraries(${TGT} PUBLIC flatbuffers-2.0)
  endfunction(FlatBuffers_Target)
endif(FlatBuffers_FOUND)
