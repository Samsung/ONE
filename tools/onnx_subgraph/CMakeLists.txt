cmake_minimum_required(VERSION 3.10)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_STANDARD 17)

project(onnx-subgraph-parser)

find_package(Protobuf REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(PROTO_FILES onnx.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Python3_INCLUDE_DIRS})

file(GLOB SOURCES "src/lib/*.cpp" )

add_library(onnx-subgraph-parser STATIC ${SOURCES} ${PROTO_SRCS} ${PROTO_FILES})
target_link_libraries(onnx-subgraph-parser protobuf jsoncpp)

add_executable(onnx-subgraph src/main.cpp)
target_link_libraries(onnx-subgraph onnx-subgraph-parser)
target_link_libraries(onnx-subgraph  ${Python3_LIBRARIES})

set(ONNX_SUBGRAPH_FILES
    test_model_download.sh
    extract_onnx.py
    subgraphs_ios.txt
    single_vs_multiple_onnx.py
)
foreach(ONNX_SUBGRAPH IN ITEMS ${ONNX_SUBGRAPH_FILES})
    set(ONNX_SUBGRAPH_FILE ${ONNX_SUBGRAPH})
    set(ONNX_SUBGRAPH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ONNX_SUBGRAPH_FILE}")
    set(ONNX_SUBGRAPH_BIN "${CMAKE_CURRENT_BINARY_DIR}/scripts/${ONNX_SUBGRAPH_FILE}")
    set(ONNX_SUBGRAPH_TARGET "${ONNX_SUBGRAPH}_target")

    add_custom_command(OUTPUT ${ONNX_SUBGRAPH_BIN}
        COMMAND ${CMAKE_COMMAND} -E copy "${ONNX_SUBGRAPH_SRC}" "${ONNX_SUBGRAPH_BIN}"
        DEPENDS ${ONNX_SUBGRAPH_SRC}
        COMMENT "Generate ${ONNX_SUBGRAPH_BIN}"
    )
    add_custom_target(${ONNX_SUBGRAPH_TARGET} ALL DEPENDS ${ONNX_SUBGRAPH_BIN})
    install(FILES ${ONNX_SUBGRAPH_BIN}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        DESTINATION bin
    )
endforeach(ONNX_SUBGRAPH)
