if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

file(GLOB_RECURSE SOURCES "*.cc")

nnfw_find_package(glib-2.0 REQUIRED)

add_executable(npud ${SOURCES})
set_target_properties(npud PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud PRIVATE glib-2.0)
install(TARGETS npud DESTINATION bin)
