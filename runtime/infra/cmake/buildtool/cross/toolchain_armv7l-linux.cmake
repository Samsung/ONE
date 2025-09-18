#
# config for arm-linux
#

# Set CMAKE_SYSTEM_NAME to notify to cmake that we are cross compiling
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

if(DEFINED ENV{USE_CLANG} AND "$ENV{USE_CLANG}" STREQUAL "1")
  set(CMAKE_C_COMPILER  clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_C_COMPILER_TARGET   arm-linux-gnueabihf)
  set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)
else()
  set(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc)
  set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
endif()

# where is the target environment
if(DEFINED ENV{ROOTFS_DIR})
  set(ROOTFS_DIR $ENV{ROOTFS_DIR})
else()
  set(ROOTFS_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../tools/cross/rootfs/arm")
endif()
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
