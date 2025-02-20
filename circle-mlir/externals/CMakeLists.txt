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
set(LP_BUILD_DIR   "${EXTERNALS_BUILD_INST_DIR}/llvm-project-build")
set(LP_INSTALL_DIR "${EXTERNALS_BUILD_INST_DIR}/llvm-project-install")

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

# build MLIR of LLVM
# from https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md
ExternalProject_Add(externals-llvm-project
  GIT_REPOSITORY "https://github.com/llvm/llvm-project.git"
  GIT_TAG        "91088978d712cd7b33610c59f69d87d5a39e3113"
  SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project"
  SOURCE_SUBDIR  "llvm"
  BINARY_DIR     "${LP_BUILD_DIR}"
  INSTALL_DIR    "${LP_INSTALL_DIR}"
  CMAKE_ARGS     -DLLVM_ENABLE_PROJECTS=mlir
                 -DLLVM_TARGETS_TO_BUILD=host
                 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX='${LP_INSTALL_DIR}'
                 -DLLVM_ENABLE_ASSERTIONS=ON
                 -DLLVM_ENABLE_RTTI=ON
  # NOTE externals-flatbuffers externals-abseil-cpp are added to DEPENDS only
  #      to build in sequence and actually do not dependent on build.
  DEPENDS        externals-flatbuffers externals-abseil-cpp
)
