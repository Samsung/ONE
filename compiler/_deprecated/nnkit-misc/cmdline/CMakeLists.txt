file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnkit_support_cmdline STATIC ${SOURCES})
target_include_directories(nnkit_support_cmdline PUBLIC include)
target_link_libraries(nnkit_support_cmdline PUBLIC nnkit_intf_cmdline)
