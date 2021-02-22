file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnkit_support_tf-1.13 STATIC ${SOURCES})
set_target_properties(nnkit_support_tf-1.13 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnkit_support_tf-1.13 PUBLIC include)
target_link_libraries(nnkit_support_tf-1.13 nnkit_intf_backend nnkit_support_tftestinfo)
target_link_libraries(nnkit_support_tf-1.13 tensorflow-1.13)

add_library(nnkit_support_tf ALIAS nnkit_support_tf-1.13)
