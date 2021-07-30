add_library(bulk_loader INTERFACE)
target_include_directories(bulk_loader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(bulk_loader INTERFACE onert_core)
