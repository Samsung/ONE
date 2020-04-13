file(GLOB_RECURSE interp_src ./*.cpp ./*.h)
add_library(mir_interpreter SHARED ${interp_src})
target_link_libraries(mir_interpreter PUBLIC mir)
target_include_directories(mir_interpreter PUBLIC include)
