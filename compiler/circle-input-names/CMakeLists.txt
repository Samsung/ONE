file(GLOB SOURCES "src/*.cpp")

add_executable(circle_input_names ${SOURCES})
target_include_directories(circle_input_names PUBLIC include)
target_link_libraries(circle_input_names PRIVATE luci_logex)
target_link_libraries(circle_input_names PRIVATE luci_lang)
target_link_libraries(circle_input_names PRIVATE crew)
install(TARGETS circle_input_names DESTINATION bin)
