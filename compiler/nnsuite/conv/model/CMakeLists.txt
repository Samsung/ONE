file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnsuite_conv STATIC ${SOURCES})
set_target_properties(nnsuite_conv PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnsuite_conv PUBLIC include)
target_link_libraries(nnsuite_conv PUBLIC angkor)
