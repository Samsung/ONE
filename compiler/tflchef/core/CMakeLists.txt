file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(tflchef_core STATIC ${SOURCES})
target_include_directories(tflchef_core PUBLIC include)
target_include_directories(tflchef_core PRIVATE src)
target_link_libraries(tflchef_core tflchef_proto)
target_link_libraries(tflchef_core tflchef_log)
target_link_libraries(tflchef_core mio_tflite2121)
target_link_libraries(tflchef_core souschef)
