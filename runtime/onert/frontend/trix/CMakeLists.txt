if (NOT BUILD_TRIX_LOADER)
    return()
endif ()

set(TRIX_LOADER_SOURCES src/trix_loader.cc)

add_library(trix_loader STATIC ${TRIX_LOADER_SOURCES})
set_target_properties(trix_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(trix_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(trix_loader PRIVATE onert_core)
target_link_libraries(trix_loader PRIVATE nnfw_common nnfw_coverage)
