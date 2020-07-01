file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(circlechef_core STATIC ${SOURCES})
target_include_directories(circlechef_core PUBLIC include)
target_include_directories(circlechef_core PRIVATE src)
target_link_libraries(circlechef_core circlechef_proto)
target_link_libraries(circlechef_core circlechef_log)
target_link_libraries(circlechef_core mio_circle)
target_link_libraries(circlechef_core souschef)
