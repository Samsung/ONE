#
# config for arm-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# where is the target environment
set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../..")
set(ROOTFS_ARM "${NNAS_PROJECT_SOURCE_DIR}/tools/cross/rootfs/arm")
include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/OptionTools.cmake")

envoption(ROOTFS_DIR ${ROOTFS_ARM})
if(NOT EXISTS "${ROOTFS_DIR}/lib/arm-linux-gnueabihf")
  message(FATAL_ERROR "Please prepare RootFS for ARM")
endif()

set(CMAKE_SYSROOT ${ROOTFS_DIR})
set(CMAKE_FIND_ROOT_PATH ${ROOTFS_DIR})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# for python binding

set(PYTHON_EXECUTABLE ${ROOTFS_ARM}/usr/bin/python3.8)

execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${ROOTFS_ARM}/usr/lib/ld-linux-armhf.so.3
                /lib/ld-linux-armhf.so.3
        RESULT_VARIABLE symlink_result
        OUTPUT_VARIABLE symlink_output
        ERROR_VARIABLE symlink_error
)

set(ENV{LD_LIBRARY_PATH} "${ROOTFS_ARM}/usr/lib/arm-linux-gnueabihf/:$ENV{LD_LIBRARY_PATH}")

# Set cache variable to ignore try-run error by find_package(Threads REQUIRED) on cross build
set(THREADS_PTHREAD_ARG "2" CACHE STRING "Result from TRY_RUN" FORCE)
