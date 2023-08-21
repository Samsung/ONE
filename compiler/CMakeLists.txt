# get CODENAME to perform per codename actions
# set focal as default
set(ONE_UBUNTU_CODENAME "focal")
find_program(LSB_RELEASE_EXEC lsb_release)
if(LSB_RELEASE_EXEC)
  # output should be one of 'bionic', 'focal', 'jammy'
  # others are not tested
  execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename
                  OUTPUT_VARIABLE ONE_UBUNTU_CODENAME
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  message(STATUS "WARNING: lsb_release not found")
endif()

if(${ONE_UBUNTU_CODENAME} STREQUAL "bionic")
  set(ONE_UBUNTU_CODENAME_BIONIC TRUE)
endif()
if(${ONE_UBUNTU_CODENAME} STREQUAL "jammy")
  set(ONE_UBUNTU_CODENAME_JAMMY TRUE)
endif()

# TODO Validate the argument of "requires"
function(get_project_build_order VAR)
  # This file will describe the dependencies among projects
  set(DEPS_FILE "${CMAKE_CURRENT_BINARY_DIR}/compiler.deps")

  # Remove .deps file
  file(REMOVE "${DEPS_FILE}")

  # Let's create .deps file
  list_subdirectories(PROJECT_DIRS)

  foreach(PROJECT_DIR IN ITEMS ${PROJECT_DIRS})
    set(SUCC "${PROJECT_DIR}")
    set(REQUIRES_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_DIR}/requires.cmake")

    macro(require PRED)
      file(APPEND "${DEPS_FILE}" "${PRED} ${SUCC} ")
    endmacro(require)

    file(APPEND "${DEPS_FILE}" "${SUCC} ${SUCC} ")
    if(EXISTS "${REQUIRES_FILE}")
      include(${REQUIRES_FILE})
    endif(EXISTS "${REQUIRES_FILE}")
  endforeach(PROJECT_DIR)

  # NOTE "tsort" is a part of the POSIX.1 standard.
  #
  # Reference: http://pubs.opengroup.org/onlinepubs/9699919799/utilities/tsort.html
  execute_process(COMMAND tsort "${DEPS_FILE}"
                  OUTPUT_VARIABLE ORDER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Remove newline characters
  # TODO Check which one (UNIX_COMMAND or WINDOWS_COMMAND) is correct
  separate_arguments(ORDER UNIX_COMMAND ${ORDER})

  set(${VAR} "${ORDER}" PARENT_SCOPE)
endfunction(get_project_build_order)

function(add_compiler_directory DIR)
  string(TOUPPER ${DIR} PREFIX)

  option(BUILD_COMPILER_${PREFIX} "Build compiler/${dir}" ON)
  set(BUILD_WHITELIST "" CACHE STRING "Set modules to be built")

  if(NOT BUILD_WHITELIST STREQUAL "")
    set(ENABLE OFF)
    set(CURRENT_DIR ${DIR})
    foreach(ACCEPTED_DIR IN ITEMS ${BUILD_WHITELIST})
      if(ACCEPTED_DIR STREQUAL CURRENT_DIR)
        set(ENABLE ON)
      endif()
    endforeach(ACCEPTED_DIR)
  else()
    set(ENABLE ${BUILD_COMPILER_${PREFIX}})
  endif()

  # This line prevents some errors in this CMakeLists.txt
  if(NOT DEFINED ENABLE)
    message(FATAL_ERROR "Undefined ENABLE! Please check CMakeLists.txt")
  endif()

  if(ENABLE)
    message(STATUS "Configure ${PREFIX}")
    add_subdirectory(${DIR})
    message(STATUS "Configure ${PREFIX} - Done")
  endif(ENABLE)
endfunction(add_compiler_directory)

function(add_compiler_directories)
  get_project_build_order(PROJECT_DIRS)

  foreach(PROJECT_DIR IN ITEMS ${PROJECT_DIRS})
    add_compiler_directory(${PROJECT_DIR})
  endforeach(PROJECT_DIR)
endfunction(add_compiler_directories)

add_compiler_directories()
