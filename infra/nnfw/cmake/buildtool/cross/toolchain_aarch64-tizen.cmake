#
# config for aarch64-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(TIZEN_TOOLCHAIN "aarch64-tizen-linux-gnu/6.2.1")

# where is the target environment
set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../..")
set(ROOTFS_AARCH64 "${NNAS_PROJECT_SOURCE_DIR}/tools/cross/rootfs/aarch64")
include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/OptionTools.cmake")

envoption(ROOTFS_DIR ${ROOTFS_AARCH64})
if(NOT EXISTS "${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
  message(FATAL_ERROR "Please prepare RootFS for tizen aarch64")
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

add_compile_options(--sysroot=${ROOTFS_DIR})

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${ROOTFS_DIR}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${ROOTFS_DIR}")

include_directories(SYSTEM ${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}/include/c++/)
include_directories(SYSTEM ${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}/include/c++/aarch64-tizen-linux-gnu)
add_compile_options(-Wno-deprecated-declarations) # compile-time option
add_compile_options(-D__extern_always_inline=inline) # compile-time option

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -B${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_DIR}/lib64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -B${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_DIR}/lib64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
