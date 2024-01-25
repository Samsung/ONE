# Let's build and install Boost libraries
function(_Boost_Build Boost_PREFIX)
  nnas_find_package(BoostSource QUIET)

  if(NOT BoostSource_FOUND)
    return()
  endif(NOT BoostSource_FOUND)

  #### Generic configurations
  if(NOT EXISTS ${BoostSource_DIR}/b2)
    execute_process(COMMAND "${BoostSource_DIR}/bootstrap.sh"
                    WORKING_DIRECTORY ${BoostSource_DIR}
                    RESULT_VARIABLE Boost_BUILD)
  endif()

  set(BoostBuild_DIR ${CMAKE_BINARY_DIR}/externals/boost)
  set(BoostInstall_DIR ${Boost_PREFIX})

  set(INSTALL_STAMP_PATH "${BoostInstall_DIR}/BOOST.stamp")
  set(BUILD_LOG_PATH "${BoostBuild_DIR}/BOOST.log")
  set(PKG_NAME "BOOST")
  set(PKG_IDENTIFIER "1.84.0")

  if(EXISTS ${INSTALL_STAMP_PATH})
    file(READ ${INSTALL_STAMP_PATH} READ_IDENTIFIER)
    if("${READ_IDENTIFIER}" STREQUAL "${PKG_IDENTIFIER}")
      return()
    endif("${READ_IDENTIFIER}" STREQUAL "${PKG_IDENTIFIER}")
  endif(EXISTS ${INSTALL_STAMP_PATH})

  unset(Boost_Options)

  list(APPEND Boost_Options --build-dir=${BoostBuild_DIR})
  list(APPEND Boost_Options --prefix=${BoostInstall_DIR})
  list(APPEND Boost_Options --with-log)
  list(APPEND Boost_Options --with-program_options)
  list(APPEND Boost_Options --with-system)
  list(APPEND Boost_Options --with-filesystem)

  if(DEFINED EXTERNALS_BUILD_THREADS)
    set(N ${EXTERNALS_BUILD_THREADS})
  else(DEFINED EXTERNALS_BUILD_THREADS)
    include(ProcessorCount)
    ProcessorCount(N)
  endif(DEFINED EXTERNALS_BUILD_THREADS)

  if((NOT N EQUAL 0) AND BUILD_EXT_MULTITHREAD)
    list(APPEND Boost_Options -j${N})
  endif()

  set(JAM_FILENAME ${BoostBuild_DIR}/user-config.jam)

  if(ANDROID)
    set(NDK_CXX ${NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/${TARGET_ARCH}-linux-android${ANDROID_API_LEVEL}-clang++)
    file(WRITE ${JAM_FILENAME} "using clang : arm64v8a : ${NDK_CXX} ;")
    list(APPEND Boost_Options toolset=clang-arm64v8a)
    # without target-os=android, it complains it cannot find -lrt.
    list(APPEND Boost_Options target-os=android)
  else()
    file(WRITE ${JAM_FILENAME} "using gcc : local : ${CMAKE_CXX_COMPILER} ;\n")
    list(APPEND Boost_Options toolset=gcc-local)
  endif(ANDROID)

  # Install Boost libraries
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${BoostInstall_DIR}")
  execute_process(COMMAND /usr/bin/env BOOST_BUILD_PATH="${BoostBuild_DIR}" ${BoostSource_DIR}/b2 install ${Boost_Options}
                  WORKING_DIRECTORY ${BoostSource_DIR}
                  OUTPUT_FILE ${BUILD_LOG_PATH}
                  RESULT_VARIABLE BUILD_EXITCODE)

  if(NOT BUILD_EXITCODE EQUAL 0)
    message(FATAL_ERROR "${PKG_NAME} Package: Build and install failed (check '${BUILD_LOG_PATH}' for details)")
  endif(NOT BUILD_EXITCODE EQUAL 0)

  file(WRITE "${INSTALL_STAMP_PATH}" "${PKG_IDENTIFIER}")

endfunction(_Boost_Build)

# Find pre-installed boost library and update Boost variables.
if (NOT BUILD_BOOST)
  # BoostConfig.cmake does not honor QUIET argument at least till cmake 1.70.0.
  # Thus, don't try to find_package if you're not entirely sure you have boost.
  find_package(Boost 1.58.0 QUIET COMPONENTS log program_options filesystem system)
  if(Boost_FOUND)
    return()
  endif()
endif()

set(Boost_PREFIX ${EXT_OVERLAY_DIR})

if(BUILD_BOOST)
  _Boost_Build("${Boost_PREFIX}")

  # Without Boost_INCLUDE_DIR, it complains the variable is missing during find_package.
  set(Boost_INCLUDE_DIR ${Boost_PREFIX}/include)

  # 1) without static build, it will complain it cannot find libc++_shared.so.
  # 2) We uses static libraries for other libraries.
  set(Boost_USE_STATIC_LIBS ON)

  # We built boost library so update Boost variables.
  find_package(Boost 1.58.0 QUIET COMPONENTS log program_options filesystem system)
endif(BUILD_BOOST)
