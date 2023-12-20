function(ExternalBuild_CMake)
  # CMAKE_DIR   Path to cmake root script to build (required)
  # BUILD_DIR   Path to build workspace (required)
  # INSTALL_DIR Path to install (required)
  # PKG_NAME    External package name word for logging and stamp file name (required)
  # IDENTIFIER  String to identify package version (optional)
  # BUILD_FLAGS Multiple argument to set compiler flag
  # EXTRA_OPTS  Multiple argument to pass options, etc for cmake configuration
  include(CMakeParseArguments)
  cmake_parse_arguments(ARG
                        ""
                        "CMAKE_DIR;BUILD_DIR;INSTALL_DIR;PKG_NAME;IDENTIFIER"
                        "BUILD_FLAGS;EXTRA_OPTS"
                        ${ARGN}
  )

  set(BUILD_LOG_PATH "${ARG_BUILD_DIR}/${ARG_PKG_NAME}.log")
  set(INSTALL_STAMP_PATH "${ARG_INSTALL_DIR}/${ARG_PKG_NAME}.stamp")
  set(INSTALL_LOG_PATH "${ARG_INSTALL_DIR}/${ARG_PKG_NAME}.log")

  set(PKG_IDENTIFIER "")
  if(DEFINED ARG_IDENTIFIER)
    set(PKG_IDENTIFIER "${ARG_IDENTIFIER}")
  endif(DEFINED ARG_IDENTIFIER)

  # NOTE Do NOT build pre-installed exists
  if(EXISTS ${INSTALL_STAMP_PATH})
    file(READ ${INSTALL_STAMP_PATH} READ_IDENTIFIER)
    if("${READ_IDENTIFIER}" STREQUAL "${PKG_IDENTIFIER}")
      return()
    endif("${READ_IDENTIFIER}" STREQUAL "${PKG_IDENTIFIER}")
  endif(EXISTS ${INSTALL_STAMP_PATH})

  message(STATUS "Build ${ARG_PKG_NAME} from ${ARG_CMAKE_DIR}")

  # if we're doing the cross compilation, external project also needs it
  if(CMAKE_TOOLCHAIN_FILE)
    set(TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE})
    # NOTE CMAKE_TOOLCHAIN_FILE maybe relative path -> make abs folder
    if(NOT EXISTS ${TOOLCHAIN_FILE})
      set(TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/${CMAKE_TOOLCHAIN_FILE})
      if(NOT EXISTS ${TOOLCHAIN_FILE})
        message(FATAL "Failed to find ${CMAKE_TOOLCHAIN_FILE}")
      endif()
    endif()
    message(STATUS "ExternalBuild_CMake TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
    list(APPEND ARG_EXTRA_OPTS -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE})
  endif(CMAKE_TOOLCHAIN_FILE)

  file(MAKE_DIRECTORY ${ARG_BUILD_DIR})
  file(MAKE_DIRECTORY ${ARG_INSTALL_DIR})

  execute_process(COMMAND ${CMAKE_COMMAND}
                            -G "${CMAKE_GENERATOR}"
                            -DCMAKE_INSTALL_PREFIX=${ARG_INSTALL_DIR}
                            -DCMAKE_BUILD_TYPE=Release
                            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} ${ARG_BUILD_FLAGS}
                            ${ARG_EXTRA_OPTS}
                            ${ARG_CMAKE_DIR}
                  OUTPUT_FILE ${BUILD_LOG_PATH}
                  ERROR_FILE ${BUILD_LOG_PATH}
                  WORKING_DIRECTORY ${ARG_BUILD_DIR}
                  RESULT_VARIABLE BUILD_EXITCODE)

  if(NOT BUILD_EXITCODE EQUAL 0)
    message(FATAL_ERROR "${ARG_PKG_NAME} Package: Build failed (check '${BUILD_LOG_PATH}' for details)")
  endif(NOT BUILD_EXITCODE EQUAL 0)

  set(NUM_BUILD_THREADS 1)
  if(DEFINED EXTERNALS_BUILD_THREADS)
    set(NUM_BUILD_THREADS ${EXTERNALS_BUILD_THREADS})
  endif(DEFINED EXTERNALS_BUILD_THREADS)

  execute_process(COMMAND ${CMAKE_COMMAND} --build . -- -j${NUM_BUILD_THREADS} install
                  OUTPUT_FILE ${INSTALL_LOG_PATH}
                  ERROR_FILE ${INSTALL_LOG_PATH}
                  WORKING_DIRECTORY ${ARG_BUILD_DIR}
                  RESULT_VARIABLE INSTALL_EXITCODE)

  if(NOT INSTALL_EXITCODE EQUAL 0)
    message(FATAL_ERROR "${ARG_PKG_NAME} Package: Installation failed (check '${INSTALL_LOG_PATH}' for details)")
  endif(NOT INSTALL_EXITCODE EQUAL 0)

  file(WRITE "${INSTALL_STAMP_PATH}" "${PKG_IDENTIFIER}")

  message(STATUS "${ARG_PKG_NAME} Package: Done")
endfunction(ExternalBuild_CMake)
