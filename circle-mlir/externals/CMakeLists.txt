# Match the minimum required version of LLVM and MLIR
cmake_minimum_required(VERSION 3.13.4)

project(circle-mlir-externals)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{Python3_ROOT_DIR})
  message(FATAL_ERROR "Set Python3_ROOT_DIR")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(ExternalProject)

set(EXTERNALS_BUILD_INST_DIR ${CMAKE_BINARY_DIR})

set(FB_BUILD_DIR   "${EXTERNALS_BUILD_INST_DIR}/flatbuffers-build")
set(FB_INSTALL_DIR "${EXTERNALS_BUILD_INST_DIR}/flatbuffers-install")
set(AC_BUILD_DIR   "${EXTERNALS_BUILD_INST_DIR}/abseil-cpp-build")
set(AC_INSTALL_DIR "${EXTERNALS_BUILD_INST_DIR}/abseil-cpp-install")

# build flatbuffers
ExternalProject_Add(externals-flatbuffers
  GIT_REPOSITORY "https://github.com/google/flatbuffers.git"
  GIT_TAG        "a078130c878b0f997af24d532c6ade903ea7f65b"
  SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers"
  BINARY_DIR     "${FB_BUILD_DIR}"
  INSTALL_DIR    "${FB_INSTALL_DIR}"
  CMAKE_ARGS     -DCMAKE_BUILD_TYPE=Release
                 -DCMAKE_INSTALL_PREFIX='${FB_INSTALL_DIR}'
)

# build abseil-cpp
ExternalProject_Add(externals-abseil-cpp
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG        "215105818dfde3174fe799600bb0f3cae233d0bf" # 20211102.0
  SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp"
  BINARY_DIR     "${AC_BUILD_DIR}"
  INSTALL_DIR    "${AC_INSTALL_DIR}"
  CMAKE_ARGS     -DCMAKE_BUILD_TYPE=Release
                 -DCMAKE_INSTALL_PREFIX='${AC_INSTALL_DIR}'
                 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                 -DCMAKE_CXX_STANDARD=17
                 -DCMAKE_CXX_STANDARD_REQUIRED=ON
)
