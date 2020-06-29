function(_ARMCompute_Import)
  include(FindPackageHandleStandardArgs)

  list(APPEND ARMCompute_LIB_SEARCH_PATHS ${ARMCompute_PREFIX})

  find_path(INCLUDE_DIR NAMES arm_compute/core/ITensor.h PATHS ${ARMCompute_INCLUDE_SEARCH_PATHS})

  find_library(CORE_LIBRARY NAMES  	 arm_compute_core  PATHS ${ARMCompute_LIB_SEARCH_PATHS} CMAKE_FIND_ROOT_PATH_BOTH)
  find_library(RUNTIME_LIBRARY NAMES arm_compute       PATHS ${ARMCompute_LIB_SEARCH_PATHS} CMAKE_FIND_ROOT_PATH_BOTH)
  find_library(GRAPH_LIBRARY NAMES   arm_compute_graph PATHS ${ARMCompute_LIB_SEARCH_PATHS} CMAKE_FIND_ROOT_PATH_BOTH)

  message(STATUS "Search acl in ${ARMCompute_LIB_SEARCH_PATHS}")

  if(NOT INCLUDE_DIR)
    nnas_find_package(ARMComputeSource QUIET)
    if (NOT ARMComputeSource_FOUND)
      set(ARMCompute_FOUND FALSE PARENT_SCOPE)
      return()
    endif()
    set(INCLUDE_DIR ${ARMComputeSource_DIR} ${ARMComputeSource_DIR}/include)
  endif(NOT INCLUDE_DIR)

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

### Check whether library exists
function(_ARMCompute_Check VAR LIBDIR)
  set(FOUND TRUE)

  if(NOT EXISTS "${LIBDIR}/libarm_compute_core.so")
    set(FOUND FALSE)
  endif()

  if(NOT EXISTS "${LIBDIR}/libarm_compute.so")
    set(FOUND FALSE)
  endif()

  if(NOT EXISTS "${LIBDIR}/libarm_compute_graph.so")
    set(FOUND FALSE)
  endif()

  set(${VAR} ${FOUND} PARENT_SCOPE)
endfunction(_ARMCompute_Check)

# Let's build and install ARMCompute libraries
# NOTE This function silently returns on error
function(_ARMCompute_Build ARMCompute_INSTALL_PREFIX)
  ### Check whether library exists
  _ARMCompute_Check(ARMCompute_FOUND ${ARMCompute_INSTALL_PREFIX})

  if(ARMCompute_FOUND)
    return()
  endif(ARMCompute_FOUND)

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

  if(CMAKE_BUILD_TYPE)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" SCON_BUILD_TYPE)
  else(CMAKE_BUILD_TYPE)
    set(SCON_BUILD_TYPE "release")
  endif(CMAKE_BUILD_TYPE)

  #### Architecture-specific configurations
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

  if(DEFINED ACL_BUILD_THREADS)
    set(N ${ACL_BUILD_THREADS})
  else(DEFINED ACL_BUILD_THREADS)
    include(ProcessorCount)
    ProcessorCount(N)
  endif(DEFINED ACL_BUILD_THREADS)

  if((NOT N EQUAL 0) AND BUILD_EXT_MULTITHREAD)
    list(APPEND SCONS_OPTIONS -j${N})
  endif()
  if(DEFINED BUILD_ARCH)
    list(APPEND SCONS_OPTIONS "arch=${BUILD_ARCH}")
  endif(DEFINED BUILD_ARCH)

  if(DEFINED BUILD_DIR)
    list(APPEND SCONS_OPTIONS "build_dir=${BUILD_DIR}")
  endif(DEFINED BUILD_DIR)

  message(STATUS "Build ARMCompute with ${SCONS_PATH} ('${SCONS_OPTIONS}'")

  # Build ARMCompute libraries with SCONS
  # NOTE ARMCompute SConstruct unconditioanlly appends "arm-linux-gnueabihf-" prefix for linux
  execute_process(COMMAND /usr/bin/env CC=gcc CXX=g++ "${SCONS_PATH}" ${SCONS_OPTIONS}
                  WORKING_DIRECTORY ${ARMComputeSource_DIR}
                  RESULT_VARIABLE ARMCompute_BUILD)

  # Install ARMCompute libraries to overlay
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${ARMCompute_INSTALL_PREFIX}"
                  WORKING_DIRECTORY ${ARMComputeSource_DIR}
                  RESULT_VARIABLE ARMCompute_BUILD)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy "build/${BUILD_DIR}/libarm_compute_core.so" "${ARMCompute_INSTALL_PREFIX}"
                  COMMAND ${CMAKE_COMMAND} -E copy "build/${BUILD_DIR}/libarm_compute.so" "${ARMCompute_INSTALL_PREFIX}"
                  COMMAND ${CMAKE_COMMAND} -E copy "build/${BUILD_DIR}/libarm_compute_graph.so" "${ARMCompute_INSTALL_PREFIX}"
                  WORKING_DIRECTORY ${ARMComputeSource_DIR}
                  RESULT_VARIABLE ARMCompute_BUILD)
endfunction(_ARMCompute_Build)

set(ARMCompute_PREFIX ${EXT_OVERLAY_DIR}/lib)
if(BUILD_ARMCOMPUTE)
  _ARMCompute_Build("${ARMCompute_PREFIX}")
endif(BUILD_ARMCOMPUTE)
_ARMCompute_Import()
