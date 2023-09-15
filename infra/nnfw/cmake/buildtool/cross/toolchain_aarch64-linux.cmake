#
# config for aarch64-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# where is the target environment
set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../..")
set(ROOTFS_AARCH64 "${NNAS_PROJECT_SOURCE_DIR}/tools/cross/rootfs/aarch64")
include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/OptionTools.cmake")

envoption(ROOTFS_DIR ${ROOTFS_AARCH64})
if(NOT EXISTS "${ROOTFS_DIR}/lib/aarch64-linux-gnu")
  message(FATAL_ERROR "Please prepare RootFS for AARCH64")
endif()

set(CMAKE_SYSROOT ${ROOTFS_DIR})
set(CMAKE_FIND_ROOT_PATH ${ROOTFS_DIR})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# for python binding
set(PYTHON_EXECUTABLE ${ROOTFS_AARCH64}/usr/bin/python3.8)

execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${ROOTFS_AARCH64}/usr/lib/ld-linux-aarch64.so.1
                /lib/ld-linux-aarch64.so.1
        RESULT_VARIABLE symlink_result
        OUTPUT_VARIABLE symlink_output
        ERROR_VARIABLE symlink_error
)

set(ENV{LD_LIBRARY_PATH} "${ROOTFS_AARCH64}/usr/lib/aarch64-linux-gnu/:$ENV{LD_LIBRARY_PATH}")

# Set cache variable to ignore try-run error by find_package(Threads REQUIRED) on cross build
set(THREADS_PTHREAD_ARG "2" CACHE STRING "Result from TRY_RUN" FORCE)
