# TODO Find how to test logging framework
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(moco_log SHARED ${SOURCES})
target_include_directories(moco_log PUBLIC include)
target_link_libraries(moco_log PUBLIC hermes)
target_link_libraries(moco_log PRIVATE hermes_std)
install(TARGETS moco_log DESTINATION lib)
