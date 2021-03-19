file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnfw_lib_rua_anchor STATIC ${SOURCES})
set_target_properties(nnfw_lib_rua_anchor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_lib_rua_anchor PUBLIC include)
target_link_libraries(nnfw_lib_rua_anchor PUBLIC nnfw_lib_rua_core)
target_link_libraries(nnfw_lib_rua_anchor PRIVATE nnfw_lib_rua_dyn)
target_link_libraries(nnfw_lib_rua_anchor PRIVATE nnfw_common)
