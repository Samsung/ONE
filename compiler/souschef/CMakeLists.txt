nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  message(STATUS "Build souschef: FAILED (missing Protobuf")
  return()
endif(NOT Protobuf_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(souschef STATIC ${SOURCES})
set_target_properties(souschef PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(souschef PUBLIC include)
target_link_libraries(souschef PUBLIC libprotobuf)
