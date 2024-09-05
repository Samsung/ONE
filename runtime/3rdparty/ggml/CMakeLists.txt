file(GLOB_RECURSE GGML_SRCS "src/*.c")

add_library(nnfw_ggml STATIC ${GGML_SRCS})
target_include_directories(nnfw_ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nnfw_ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
