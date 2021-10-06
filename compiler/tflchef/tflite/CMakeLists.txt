file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(tflchef_tflite STATIC ${SOURCES})
target_include_directories(tflchef_tflite PUBLIC include)
target_include_directories(tflchef_tflite PRIVATE src)
target_link_libraries(tflchef_tflite tflchef_proto)
target_link_libraries(tflchef_tflite mio_tflite260)
target_link_libraries(tflchef_tflite cwrap)
target_link_libraries(tflchef_tflite souschef)
