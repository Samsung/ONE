file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnfw_lib_profiling STATIC ${SOURCES})
set_property(TARGET nnfw_lib_profiling PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_lib_profiling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nnfw_lib_profiling PRIVATE nnfw_common)
