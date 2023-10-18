function(_ARMCompute_Import)
  include(FindPackageHandleStandardArgs)

  list(APPEND ARMCompute_LIB_SEARCH_PATHS ${ARMCompute_PREFIX}/lib)

  find_path(INCLUDE_DIR NAMES arm_compute/core/ITensor.h PATHS ${ARMCompute_INCLUDE_SEARCH_PATHS})

  find_library(CORE_LIBRARY NAMES  	 arm_compute_core  PATHS ${ARMCompute_LIB_SEARCH_PATHS} CMAKE_FIND_ROOT_PATH_BOTH)
  find_library(RUNTIME_LIBRARY NAMES arm_compute       PATHS ${ARMCompute_LIB_SEARCH_PATHS} CMAKE_FIND_ROOT_PATH_BOTH)
  find_library(GRAPH_LIBRARY NAMES   arm_compute_graph PATHS ${ARMCompute_LIB_SEARCH_PATHS} CMAKE_FIND_ROOT_PATH_BOTH)

  message(STATUS "Search acl in ${ARMCompute_LIB_SEARCH_PATHS}")

  # ARMCompute v21.02 moves some headers into "src/".
  # And we cannot build armcompute-ex library without these headers.
  # So we need to download and use source code if our build root doesn't have headers in "src/" (tizen's devel package includes these headers).
  # TODO Don't use headers in "src/"
  find_path(HEADER_SRC_DIR NAMES src/core/CL/ICLKernel.h PATHS ${ARMCompute_INCLUDE_SEARCH_PATHS})
  if(NOT INCLUDE_DIR OR NOT HEADER_SRC_DIR)
    nnas_find_package(ARMComputeSource QUIET)
    if (NOT ARMComputeSource_FOUND)
      set(ARMCompute_FOUND FALSE PARENT_SCOPE)
      return()
    endif()

    # Clean if INCLUDE_DIR is NOT_FOUND
    if(NOT INCLUDE_DIR)
      unset(INCLUDE_DIR)
    endif(NOT INCLUDE_DIR)

    list(APPEND INCLUDE_DIR ${ARMComputeSource_DIR} ${ARMComputeSource_DIR}/include)
  endif(NOT INCLUDE_DIR OR NOT HEADER_SRC_DIR)

  if(NOT CORE_LIBRARY)
    set(ARMCompute_FOUND FALSE PARENT_SCOPE)
    message(STATUS "Cannot find libarm_compute_core.so")
    return()
  endif()

  if(NOT RUNTIME_LIBRARY)
    message(STATUS "Cannot find libarm_compute.so")
    set(ARMCompute_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  if(NOT GRAPH_LIBRARY)
    message(STATUS "Cannot find libarm_compute_graph.so")
    set(ARMCompute_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  if(NOT TARGET arm_compute_core)
    add_library(arm_compute_core INTERFACE)
    target_include_directories(arm_compute_core SYSTEM INTERFACE ${INCLUDE_DIR})
	target_link_libraries(arm_compute_core INTERFACE dl ${LIB_PTHREAD})
    target_link_libraries(arm_compute_core INTERFACE ${CORE_LIBRARY})
  endif(NOT TARGET arm_compute_core)

  if(NOT TARGET arm_compute)
    add_library(arm_compute INTERFACE)
    target_include_directories(arm_compute SYSTEM INTERFACE ${INCLUDE_DIR})
    target_link_libraries(arm_compute INTERFACE ${RUNTIME_LIBRARY})
    target_link_libraries(arm_compute INTERFACE arm_compute_core)
  endif(NOT TARGET arm_compute)

  if(NOT TARGET arm_compute_graph)
    add_library(arm_compute_graph INTERFACE)
    target_include_directories(arm_compute_graph SYSTEM INTERFACE ${INCLUDE_DIR})
    target_link_libraries(arm_compute_graph INTERFACE ${GRAPH_LIBRARY})
    target_link_libraries(arm_compute_graph INTERFACE arm_compute)
  endif(NOT TARGET arm_compute_graph)

  set(ARMCompute_FOUND TRUE PARENT_SCOPE)
endfunction(_ARMCompute_Import)

# Let's build and install ARMCompute libraries
function(_ARMCompute_Build ARMComputeInstall_DIR)
  set(PKG_NAME "ARMCOMPUTE")
  set(PKG_IDENTIFIER "21.02")
  set(INSTALL_STAMP_PATH "${ARMComputeInstall_DIR}/${PKG_NAME}.stamp")
  set(ARMComputeBuild_DIR "${CMAKE_BINARY_DIR}/externals/armcompute")

  if(EXISTS ${INSTALL_STAMP_PATH})
    file(READ ${INSTALL_STAMP_PATH} READ_IDENTIFIER)
    if("${READ_IDENTIFIER}" STREQUAL "${PKG_IDENTIFIER}")
      return()
    endif("${READ_IDENTIFIER}" STREQUAL "${PKG_IDENTIFIER}")
  endif(EXISTS ${INSTALL_STAMP_PATH})

  ### Let's build with SCONS
  nnas_find_package(ARMComputeSource QUIET)

  if(NOT ARMComputeSource_FOUND)
    return()
  endif(NOT ARMComputeSource_FOUND)

  find_program(SCONS_PATH scons)

  if(NOT SCONS_PATH)
    message(WARNING "SCONS NOT FOUND. Please install SCONS to build ARMCompute.")
    return()
  endif(NOT SCONS_PATH)

  if(DEBUG_ARMCOMPUTE)
    set(SCON_BUILD_TYPE "debug")
  else(DEBUG_ARMCOMPUTE)
    set(SCON_BUILD_TYPE "release")
  endif(DEBUG_ARMCOMPUTE)

  #### Architecture-specific configurations

  #### BUILD_DIR is in source tree to reduce CI build overhead
  #### TODO Change BUILD_DIR to ${ARMComputeBuild_DIR}
  if(TARGET_ARCH STREQUAL "armv7l")
    set(BUILD_ARCH "armv7a")
    set(BUILD_DIR "${BUILD_ARCH}-${TARGET_OS}.${SCON_BUILD_TYPE}")
  endif()

  if(TARGET_ARCH STREQUAL "aarch64")
    set(BUILD_ARCH "arm64-v8a")
    set(BUILD_DIR "${BUILD_ARCH}-${TARGET_OS}.${SCON_BUILD_TYPE}")
  endif()

  #### Platform-specific configurations
  #### TODO Support android

  #### Mode-specific configurations
  if(SCON_BUILD_TYPE STREQUAL "debug")
    list(APPEND SCONS_OPTIONS "debug=1")
  endif()

  #### Generic configurations
  list(APPEND SCONS_OPTIONS "neon=1")
  list(APPEND SCONS_OPTIONS "opencl=1")
  list(APPEND SCONS_OPTIONS "examples=0")
  list(APPEND SCONS_OPTIONS "Werror=0")
  list(APPEND SCONS_OPTIONS "os=${TARGET_OS}")

  #### Disable test build
  list(APPEND SCONS_OPTIONS "benchmark_tests=0")
  list(APPEND SCONS_OPTIONS "validation_tests=0")
  list(APPEND SCONS_OPTIONS "benchmark_examples=0")
  list(APPEND SCONS_OPTIONS "validate_examples=0")
  list(APPEND SCONS_OPTIONS "reference_openmp=0")

  if(DEFINED EXTERNALS_BUILD_THREADS)
    set(N ${EXTERNALS_BUILD_THREADS})
  else(DEFINED EXTERNALS_BUILD_THREADS)
    include(ProcessorCount)
    ProcessorCount(N)
  endif(DEFINED EXTERNALS_BUILD_THREADS)

  if((NOT N EQUAL 0) AND BUILD_EXT_MULTITHREAD)
    list(APPEND SCONS_OPTIONS -j${N})
  endif()
  if(DEFINED BUILD_ARCH)
    list(APPEND SCONS_OPTIONS "arch=${BUILD_ARCH}")
  endif(DEFINED BUILD_ARCH)

  if(DEFINED BUILD_DIR)
    list(APPEND SCONS_OPTIONS "build_dir=${BUILD_DIR}")
  endif(DEFINED BUILD_DIR)

  list(APPEND SCONS_OPTIONS "install_dir=${ARMComputeInstall_DIR}")

  set(SCONS_CC "gcc")
  set(SCONS_CXX "g++")
  if(ANDROID)
    list(APPEND SCONS_OPTIONS "toolchain_prefix=${ANDROID_TOOLCHAIN_PREFIX}")
    list(APPEND SCONS_OPTIONS "compiler_prefix=${ANDROID_TOOLCHAIN_ROOT}/bin/aarch64-linux-android${ANDROID_API_LEVEL}-")
    set(SCONS_CC "clang")
    set(SCONS_CXX "clang++")
  endif(ANDROID)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    list(APPEND SCONS_OPTIONS "extra_cxx_flags=-Wno-deprecated-copy")
  endif()

  message(STATUS "Build ARMCompute with ${SCONS_PATH} ('${SCONS_OPTIONS}'")

  # Build ARMCompute libraries with SCONS
  # NOTE ARMCompute build process don't allow logging by using OUTPUT_FILE and ERROR_FILE option
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${ARMComputeInstall_DIR}")
  execute_process(COMMAND /usr/bin/env CC=${SCONS_CC} CXX=${SCONS_CXX} "${SCONS_PATH}" ${SCONS_OPTIONS}
                  WORKING_DIRECTORY ${ARMComputeSource_DIR}
                  RESULT_VARIABLE BUILD_EXITCODE)

  if(NOT BUILD_EXITCODE EQUAL 0)
    message(FATAL_ERROR "${PKG_NAME} Package: Build and install failed (check '${BUILD_LOG_PATH}' for details)")
  endif(NOT BUILD_EXITCODE EQUAL 0)

  file(WRITE "${INSTALL_STAMP_PATH}" "${PKG_IDENTIFIER}")
endfunction(_ARMCompute_Build)

set(ARMCompute_PREFIX ${EXT_OVERLAY_DIR})
if(BUILD_ARMCOMPUTE)
  _ARMCompute_Build("${ARMCompute_PREFIX}")
endif(BUILD_ARMCOMPUTE)
_ARMCompute_Import()
