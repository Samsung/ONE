message(STATUS "Build mlapse-tfl: TRUE")

file(GLOB_RECURSE SOURCES "*.cc")

add_executable(mlapse-tfl ${SOURCES})
target_include_directories(mlapse-tfl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlapse-tfl nnfw_lib_tflite)
target_link_libraries(mlapse-tfl nnfw_lib_benchmark)
target_link_libraries(mlapse-tfl tensorflow-lite)

install(TARGETS mlapse-tfl DESTINATION bin)
