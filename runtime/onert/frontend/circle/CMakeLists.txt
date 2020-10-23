if (NOT BUILD_CIRCLE_LOADER)
    return()
endif ()

set(CIRCLE_LOADER_SOURCES src/circle_loader.cc)

add_library(circle_loader SHARED ${CIRCLE_LOADER_SOURCES})

target_include_directories(circle_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(circle_loader PRIVATE onert_core)
target_link_libraries(circle_loader PRIVATE base_loader nnfw_common nnfw_coverage)
target_link_libraries(circle_loader PRIVATE circle_schema)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET circle_loader POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:circle_loader>)
endif()

install(TARGETS circle_loader DESTINATION lib)
