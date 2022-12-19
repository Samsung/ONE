file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-mpqsolver "${SOURCES}")
target_link_libraries(circle-mpqsolver arser)
target_link_libraries(circle-mpqsolver vconone)
target_link_libraries(circle-mpqsolver safemain)

install(TARGETS circle-mpqsolver DESTINATION bin)
