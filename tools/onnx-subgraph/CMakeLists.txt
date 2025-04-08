# cmake version dependency
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

file(GLOB SOURCES "src/lib/*.cpp" "src/lib/*.cpp" )

add_library(onnx-subgraph-parser STATIC ${SOURCES} ${PROTO_SRCS} ${PROTO_FILES})
target_link_libraries(onnx-subgraph-parser protobuf jsoncpp)

add_executable(onnx-subgraph src/main.cpp)
target_link_libraries(onnx-subgraph onnx-subgraph-parser ${Python3_LIBRARIES})

 set(ONNX_SUGRAPH_FILES                                                          
    extract_onnx_lib.py
    extract_onnx.py
    single_vs_multiple_onnx.py
    quant.py
    model_inference.py
    model_inference_multiple_output.py
    onnx_subgraph_ut.py
    test_model_download.sh
    config.json
    config-sample-1.json
    config-sample-2.json
 )                                                                            
                                                                              
 foreach(ONNX_SUGRAPH IN ITEMS ${ONNX_SUGRAPH_FILES})                                                                                                        
   set(ONNX_SUGRAPH_FILE ${ONNX_SUGRAPH})                                           
   set(ONNX_SUGRAPH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ONNX_SUGRAPH_FILE}")         
   set(ONNX_SUGRAPH_BIN "${CMAKE_CURRENT_BINARY_DIR}/scripts/${ONNX_SUGRAPH_FILE}")         
   set(ONNX_SUGRAPH_TARGET "${ONNX_SUGRAPH}_target")                                
                                                                              
   add_custom_command(OUTPUT ${ONNX_SUGRAPH_BIN}                                 
     COMMAND ${CMAKE_COMMAND} -E copy "${ONNX_SUGRAPH_SRC}" "${ONNX_SUGRAPH_BIN}"   
     DEPENDS ${ONNX_SUGRAPH_SRC}                                                 
     COMMENT "Generate ${ONNX_SUGRAPH_BIN}"                                      
   )                                                                          
                                                                              
   add_custom_target(${ONNX_SUGRAPH_TARGET} ALL DEPENDS ${ONNX_SUGRAPH_BIN})        
                                                                              
   install(FILES ${ONNX_SUGRAPH_BIN}                                             
           PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE                   
                       GROUP_READ GROUP_EXECUTE                               
                       WORLD_READ WORLD_EXECUTE                               
           DESTINATION bin)                                                   
                                                                              
 endforeach(ONNX_SUGRAPH)     
