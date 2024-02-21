if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

nnfw_find_package(FlatBuffers EXACT 23.5.26 QUIET)
if(NOT FlatBuffers_FOUND)
  message(STATUS "Skip build custom operation test: cannot find flatbuffers")
  return()
endif(NOT FlatBuffers_FOUND)

if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE is not specified. \
  Generated library will have `unknown` in its name, instead of `release` or `debug`.")
  set(BUILD_TYPE "unknown")
else()
  string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
endif()

set(SUFFIX ".${TARGET_ARCH}-${TARGET_OS}.${BUILD_TYPE}")

# Functions for custom op test

# Add custom op app
#
# NAME   : the app name to be built
# SOURCES: the app source codes
# KERNELS: the custom op kernel names that the app uses
function(add_nnfw_custom_op_app NAME)
  cmake_parse_arguments(
    PARSED_ARGS       # prefix of output variables
    ""                # list of names of the boolean arguments (only defined ones will be true)
    ""                # list of names of mono-valued arguments
    "SOURCES;KERNELS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN}           # arguments of the function to parse, here we take the all original ones
  )
  add_executable(${NAME} ${PARSED_ARGS_SOURCES})
  set(LIBNAMELIST "")
  foreach(KERNEL ${PARSED_ARGS_KERNELS})
    list(APPEND LIBNAMELIST "${KERNEL}${SUFFIX}")
  endforeach()
  target_link_libraries(${NAME} PRIVATE ${LIBNAMELIST})
  target_link_libraries(${NAME} PRIVATE nnfw-nnapi-header nnfw-dev)
  target_link_libraries(${NAME} PRIVATE dl ${LIB_PTHREAD})
endfunction()

# Add custom op kernel with nnpackage spec conforming name.
#
# NAME  : the custom op kernel name to be built
# STATIC: ON (or TRUE) to build static library, OFF (or FALSE) for shared library
# ARGN  : the source codes that comes after two arguments.
function(add_nnfw_custom_op_kernel NAME STATIC)
  set(LIBNAME ${NAME}${SUFFIX})
  if(STATIC)
    # message(FATAL_ERROR STATIC)
    add_library(${LIBNAME} STATIC ${ARGN})
  else()
    # message(FATAL_ERROR SHARED)
    add_library(${LIBNAME} SHARED ${ARGN})
  endif()
  target_link_libraries(${LIBNAME} PRIVATE nnfw-nnapi-header nnfw-dev)
  target_link_libraries(${LIBNAME} PRIVATE flatbuffers::flatbuffers-23.5.26)
endfunction()

# Install custom op kernel built with `add_nnfw_custom_op_kernel`
#
# NAME       : custom op kernel name
# DESTINATION: the path where the custom op kernel library will be installed in
function(install_nnfw_custom_op_kernel NAME DESTINATION)
  set(LIBNAME ${NAME}${SUFFIX})
  install(TARGETS ${LIBNAME} DESTINATION ${DESTINATION})
endfunction()

add_subdirectories()
