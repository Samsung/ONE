function(_Protobuf_module_import)
  # Let's use find_package here not to export unnecessary definitions
  find_package(Protobuf MODULE QUIET)

  if(NOT PROTOBUF_FOUND)
    set(Protobuf_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT PROTOBUF_FOUND)

  if(NOT TARGET protoc)
    add_executable(protoc IMPORTED)
    set_target_properties(protoc PROPERTIES IMPORTED_LOCATION ${PROTOBUF_PROTOC_EXECUTABLE})
   endif(NOT TARGET protoc)

  if(NOT TARGET libprotobuf)
    add_library(libprotobuf INTERFACE)
    target_include_directories(libprotobuf INTERFACE ${PROTOBUF_INCLUDE_DIRS})
    target_link_libraries(libprotobuf INTERFACE ${PROTOBUF_LIBRARIES})
  endif(NOT TARGET libprotobuf)

  set(Protobuf_FOUND TRUE PARENT_SCOPE)
endfunction(_Protobuf_module_import)

function(_Protobuf_import)
  # Let's use find_package here not to export unnecessary definitions
  # NOTE Here we use "exact" match to avoid possible infinite loop
  find_package(Protobuf EXACT 3.5.2 QUIET)

  if(NOT Protobuf_FOUND)
    set(Protobuf_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Protobuf_FOUND)

  if(NOT TARGET libprotobuf)
    add_library(libprotobuf INTERFACE)
    target_link_libraries(libprotobuf INTERFACE protobuf::libprotobuf)
  endif(NOT TARGET libprotobuf)

  set(Protobuf_FOUND TRUE PARENT_SCOPE)
endfunction(_Protobuf_import)

function(_Protobuf_build)
  if(NOT BUILD_PROTOBUF)
    return()
  endif(NOT BUILD_PROTOBUF)

  nnas_find_package(ProtobufSource QUIET)

  if(NOT ProtobufSource_FOUND)
    # Source is not available
    return()
  endif(NOT ProtobufSource_FOUND)

  # set 'EXTERNAL_JS_EMBED' environment variable
  if(NOT DEFINED ENV{EXTERNAL_JS_EMBED})
    if(DEFINED ENV{BUILD_HOST_EXEC})
      set(EXTERNAL_JS_EMBED $ENV{BUILD_HOST_EXEC}/externals/PROTOBUF/build/js_embed)
      set(ENV{EXTERNAL_JS_EMBED} ${EXTERNAL_JS_EMBED})
    endif(DEFINED ENV{BUILD_HOST_EXEC})
  endif(NOT DEFINED ENV{EXTERNAL_JS_EMBED})

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${ProtobufSource_DIR}/cmake
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/PROTOBUF/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      BUILD_FLAGS -fPIC
                      EXTRA_OPTS  -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_WITH_ZLIB=OFF
                      IDENTIFIER  "3.5.2-fix2"
                      PKG_NAME    "PROTOBUF")

endfunction(_Protobuf_build)

set(PROTOC_PATH $<TARGET_FILE:protobuf::protoc>)

if(DEFINED ENV{BUILD_HOST_EXEC})
  set(PROTOC_PATH $ENV{BUILD_HOST_EXEC}/overlay/bin/protoc)
endif(DEFINED ENV{BUILD_HOST_EXEC})
if(DEFINED ENV{EXTERNAL_PROTOC})
  set(PROTOC_PATH $ENV{EXTERNAL_PROTOC})
endif(DEFINED ENV{EXTERNAL_PROTOC})

_Protobuf_build()

if(USE_PROTOBUF_LEGACY_IMPORT)
  _Protobuf_module_import()
else(USE_PROTOBUF_LEGACY_IMPORT)
  _Protobuf_import()
endif(USE_PROTOBUF_LEGACY_IMPORT)

if(Protobuf_FOUND)
  function(Protobuf_Generate PREFIX OUTPUT_DIR PROTO_DIR)
    get_filename_component(abs_output_dir ${OUTPUT_DIR} ABSOLUTE)
    get_filename_component(abs_proto_dir ${PROTO_DIR} ABSOLUTE)

    # Let's reset variables before using them
    # NOTE This DOES NOT AFFECT variables in the parent scope
    unset(PROTO_FILES)
    unset(OUTPUT_FILES)

    foreach(proto ${ARGN})
      get_filename_component(fil "${proto}" NAME)
      get_filename_component(dir "${proto}" DIRECTORY)

      get_filename_component(fil_we "${fil}" NAME_WE)

      get_filename_component(abs_fil "${abs_proto_base}/${proto}" ABSOLUTE)
      get_filename_component(abs_dir "${abs_fil}" DIRECTORY)

      list(APPEND PROTO_FILES "${abs_proto_dir}/${proto}")
      list(APPEND OUTPUT_FILES "${abs_output_dir}/${dir}/${fil_we}.pb.h")
      list(APPEND OUTPUT_FILES "${abs_output_dir}/${dir}/${fil_we}.pb.cc")
    endforeach()

    add_custom_command(OUTPUT ${OUTPUT_FILES}
                       COMMAND ${CMAKE_COMMAND} -E make_directory "${abs_output_dir}"
                       COMMAND "${PROTOC_PATH}" --cpp_out "${abs_output_dir}" -I "${abs_proto_dir}" ${PROTO_FILES}
                       DEPENDS ${PROTO_FILES})

    set(${PREFIX}_SOURCES ${OUTPUT_FILES} PARENT_SCOPE)
    set(${PREFIX}_INCLUDE_DIRS ${abs_output_dir} PARENT_SCOPE)
    set(${PREFIX}_LIBRARIES protobuf::libprotobuf PARENT_SCOPE)
  endfunction(Protobuf_Generate)
endif(Protobuf_FOUND)
