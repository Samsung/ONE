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

set(ONNX_SUGRAPH_FILES                                                          
    test_model_download.sh
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
