#
# config for arm-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER   arm-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabi-g++)

set(TIZEN_TOOLCHAIN "armv7l-tizen-linux-gnueabi/6.2.1")

# where is the target environment
set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../..")
set(ROOTFS_ARM "${NNAS_PROJECT_SOURCE_DIR}/tools/cross/rootfs/armel")
include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/OptionTools.cmake")

envoption(ROOTFS_DIR ${ROOTFS_ARM})
if(NOT EXISTS "${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
  message(FATAL_ERROR "Please prepare RootFS for tizen ARM softfp")
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


add_compile_options(-mthumb)
add_compile_options(-mfpu=neon-vfpv4)
add_compile_options(-mfloat-abi=softfp)
add_compile_options(--sysroot=${ROOTFS_DIR})

include_directories(SYSTEM ${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}/include/c++/)
include_directories(SYSTEM ${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}/include/c++/armv7l-tizen-linux-gnueabi)
add_compile_options(-Wno-deprecated-declarations) # compile-time option
add_compile_options(-D__extern_always_inline=inline) # compile-time option

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -B${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_DIR}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -B${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_DIR}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_DIR}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
