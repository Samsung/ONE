function(_TensorFlowProtoText_import)
  macro(require_package PKGNAME)
    nnas_find_package(${PKGNAME} ${ARGN} QUIET)
    if(NOT ${PKGNAME}_FOUND)
      message(STATUS "Found TensorFlowProtoText: FALSE (${PKGNAME} is missing)")
      set(TensorFlowProtoText_FOUND FALSE PARENT_SCOPE)
      return()
    endif(NOT ${PKGNAME}_FOUND)
  endmacro(require_package)

  require_package(TensorFlowSource EXACT 1.12)
  require_package(Abseil)
  require_package(Eigen-fd6845384b86)
  require_package(Protobuf)
  require_package(GoogleDoubleConversion)
  require_package(GoogleNSync)

  if(NOT TARGET tensorflow-prototext-1.12)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/build" TensorFlowProtoText-1.12)
  endif(NOT TARGET tensorflow-prototext-1.12)

  set(TensorFlowProtoText_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowProtoText_import)

_TensorFlowProtoText_import()

if(TensorFlowProtoText_FOUND)
  # CMAKE_CURRENT_LIST_DIR
  #
  # ... The value has dynamic scope. ... Therefore the value of the variable inside a macro
  # or function is the directory of the file invoking the bottom-most entry on the call stack,
  # not the directory of the file containing the macro or function definition.
  #
  # Reference: https://cmake.org/cmake/help/v3.1/variable/CMAKE_CURRENT_LIST_DIR.html
  set(TENSORLFLOW_PROTO_TEXT_1_12_CMAKE_DIR
    "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL
    "Where to find make_directories"
  )

  # Comments from "gen_proto_text_functions.cc"
  # >
  # > Main program to take input protos and write output pb_text source files that
  # > contain generated proto text input and output functions.
  # >
  # > Main expects:
  # > - First argument is output path
  # > - Second argument is the relative path of the protos to the root. E.g.,
  # >   for protos built by a rule in tensorflow/core, this will be
  # >   tensorflow/core.
  # > - Then any number of source proto file names, plus one source name must be
  # >   placeholder.txt from this gen tool's package.  placeholder.txt is
  # >   ignored for proto resolution, but is used to determine the root at which
  # >   the build tool has placed the source proto files.
  # >
  function(ProtoText_Generate PREFIX OUTPUT_DIR)
    # THIS SHOULD SUCCEED!
    nnas_find_package(TensorFlowSource EXACT 1.12 REQUIRED)

    set(OUTPUT_REL "tensorflow")
    set(PROTO_DIR "${TensorFlowSource_DIR}")

    set(PROTO_INPUTS ${ARGN})
    list(APPEND PROTO_INPUTS "tensorflow/tools/proto_text/placeholder.txt")

    get_filename_component(abs_output_dir ${OUTPUT_DIR} ABSOLUTE)
    get_filename_component(abs_proto_dir ${TensorFlowSource_DIR} ABSOLUTE)

    # Let's reset variables before using them
    # NOTE This DOES NOT AFFECT variables in the parent scope
    unset(PROTO_FILES)
    unset(OUTPUT_DIRS)
    unset(OUTPUT_FILES)

    foreach(proto ${PROTO_INPUTS})
      get_filename_component(fil "${proto}" NAME)
      get_filename_component(dir "${proto}" DIRECTORY)

      get_filename_component(fil_we "${fil}" NAME_WE)

      get_filename_component(abs_fil "${abs_proto_base}/${proto}" ABSOLUTE)
      get_filename_component(abs_dir "${abs_fil}" DIRECTORY)

      list(APPEND PROTO_FILES "${abs_proto_dir}/${proto}")

      if(NOT ${fil} STREQUAL "placeholder.txt")
        list(APPEND OUTPUT_DIRS "${abs_output_dir}/${dir}")
        list(APPEND OUTPUT_FILES "${abs_output_dir}/${dir}/${fil_we}.pb_text.h")
        list(APPEND OUTPUT_FILES "${abs_output_dir}/${dir}/${fil_we}.pb_text-impl.h")
        list(APPEND OUTPUT_FILES "${abs_output_dir}/${dir}/${fil_we}.pb_text.cc")
      endif(NOT ${fil} STREQUAL "placeholder.txt")
    endforeach()

    add_custom_command(OUTPUT ${OUTPUT_FILES}
      # "make_directory" in CMake 3.1 cannot create multiple directories at once.
      # COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIRS}
      COMMAND "${TENSORLFLOW_PROTO_TEXT_1_12_CMAKE_DIR}/make_directories.sh" ${OUTPUT_DIRS}
      COMMAND "$<TARGET_FILE:tensorflow-prototext-1.12>" "${abs_output_dir}/${OUTPUT_REL}" "${OUTPUT_REL}" ${PROTO_FILES}
      DEPENDS ${PROTO_FILES})

    set(${PREFIX}_SOURCES ${OUTPUT_FILES} PARENT_SCOPE)
    set(${PREFIX}_INCLUDE_DIRS ${abs_output_dir} PARENT_SCOPE)
  endfunction(ProtoText_Generate)
endif(TensorFlowProtoText_FOUND)
