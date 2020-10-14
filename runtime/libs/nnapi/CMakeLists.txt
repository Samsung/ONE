add_library(nnfw_lib_nnapi INTERFACE)

target_include_directories(nnfw_lib_nnapi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nnfw_lib_nnapi INTERFACE nnfw-nnapi-header)
