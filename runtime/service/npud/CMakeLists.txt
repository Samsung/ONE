if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

nnfw_find_package(GLib2.0 REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_executable(npud ${SOURCES})
set_target_properties(npud PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud PRIVATE ${LIB_PTHREAD})

install(TARGETS npud DESTINATION bin)
