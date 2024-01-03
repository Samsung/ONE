if(NOT BUILD_CIRCLE_LOADER)
    return()
endif()

nnfw_find_package(FlatBuffers REQUIRED)

set(TRAININFO_SOURCES src/traininfo_loader.cc)

add_library(traininfo_loader STATIC ${TRAININFO_SOURCES})
set_target_properties(traininfo_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(traininfo_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(traininfo_loader PRIVATE onert_core)
target_link_libraries(traininfo_loader PRIVATE flatbuffers::flatbuffers)
