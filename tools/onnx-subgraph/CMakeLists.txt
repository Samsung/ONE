# cmake version dependency
cmake_minimum_required(VERSION 3.10)

set(ONNX_SUGRAPH_FILES                                                          
    extract_onnx.py
    test_model_download.sh
    subgraphs_ios.txt
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
