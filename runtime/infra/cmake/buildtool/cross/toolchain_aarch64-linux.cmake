#
# config for aarch64-linux
#

# Set CMAKE_SYSTEM_NAME to notify to cmake that we are cross compiling
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(DEFINED ENV{USE_CLANG} AND "$ENV{USE_CLANG}" STREQUAL "1")
  set(CMAKE_C_COMPILER   clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_C_COMPILER_TARGET   arm-linux-gnueabihf)
  set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)
else()
  set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif()

# where is the target RootFS
if(DEFINED ENV{ROOTFS_DIR})
  set(ROOTFS_DIR $ENV{ROOTFS_DIR})
else()
  set(ROOTFS_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../tools/cross/rootfs/aarch64")
endif()
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

# Set cache variable to ignore try-run error by find_package(Threads REQUIRED) on cross build
set(THREADS_PTHREAD_ARG "2" CACHE STRING "Result from TRY_RUN" FORCE)
