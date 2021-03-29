file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle_partitioner "${SOURCES}")
target_link_libraries(circle_partitioner safemain)
target_link_libraries(circle_partitioner arser)
target_link_libraries(circle_partitioner vconone)

install(TARGETS circle_partitioner DESTINATION bin)
