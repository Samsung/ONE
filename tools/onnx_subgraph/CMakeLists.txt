cmake_minimum_required(VERSION 3.10)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_STANDARD 17)

project(onnx-subgraph-parser)

find_package(Protobuf REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${Python3_INCLUDE_DIRS})
