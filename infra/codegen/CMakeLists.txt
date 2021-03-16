# minimal version required by Halide
cmake_minimum_required(VERSION 3.16)

project(codegen)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE
        INTERNAL "Where to find nnas top-level source directory"
        )

# This project depends on components from compiler
include("${NNAS_PROJECT_SOURCE_DIR}/infra/nncc/CMakeLists.txt")

#add_subdirectory("${NNAS_PROJECT_SOURCE_DIR}/compiler" "${CMAKE_BINARY_DIR}/compiler")
add_subdirectory("${NNAS_PROJECT_SOURCE_DIR}/codegen" "${CMAKE_BINARY_DIR}/codegen")
