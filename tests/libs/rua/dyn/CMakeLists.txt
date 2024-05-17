file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnfw_lib_rua_dyn STATIC ${SOURCES})
set_target_properties(nnfw_lib_rua_dyn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_lib_rua_dyn PUBLIC include)
target_link_libraries(nnfw_lib_rua_dyn PUBLIC nnfw_lib_rua_core)
target_link_libraries(nnfw_lib_rua_dyn PRIVATE nnfw_common)
