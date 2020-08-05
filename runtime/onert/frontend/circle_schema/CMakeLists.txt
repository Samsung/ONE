add_library(circle_schema INTERFACE)

target_link_libraries(circle_schema INTERFACE flatbuffers::flatbuffers)

target_include_directories(circle_schema INTERFACE include)
