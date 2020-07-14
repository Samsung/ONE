file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnfw_lib_benchmark SHARED ${SOURCES})
target_include_directories(nnfw_lib_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nnfw_lib_benchmark PRIVATE ${LIB_PTHREAD})
install(TARGETS nnfw_lib_benchmark DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/nnfw)
