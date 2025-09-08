add_library(circle_schema INTERFACE)

nnfw_find_package(FlatBuffers EXACT 23.5.26 REQUIRED)

target_link_libraries(circle_schema INTERFACE flatbuffers::flatbuffers)

target_include_directories(circle_schema INTERFACE include)
