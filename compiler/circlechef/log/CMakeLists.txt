# TODO Find how to test logging framework
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(circlechef_log STATIC ${SOURCES})
target_include_directories(circlechef_log PUBLIC include)
target_link_libraries(circlechef_log PUBLIC hermes)
target_link_libraries(circlechef_log PRIVATE hermes_std)
