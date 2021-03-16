set(SOURCES src/Filesystem.cpp include/Filesystem.h)

add_library(luci_codegen_filesystem STATIC ${SOURCES})

set_property(TARGET luci_codegen_filesystem PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(luci_codegen_filesystem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
