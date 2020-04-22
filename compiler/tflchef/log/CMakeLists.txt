# TODO Find how to test logging framework
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(tflchef_log STATIC ${SOURCES})
target_include_directories(tflchef_log PUBLIC include)
target_link_libraries(tflchef_log PUBLIC hermes)
target_link_libraries(tflchef_log PRIVATE hermes_std)
