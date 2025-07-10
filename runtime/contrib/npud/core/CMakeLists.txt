nnfw_find_package(GLib2.0 REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE MAIN_SOURCE_FILE "main.cc")
list(REMOVE_ITEM SOURCES ${MAIN_SOURCE_FILE})

add_library(npud_core STATIC ${SOURCES})

set_target_properties(npud_core PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud_core PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_include_directories(npud_core PUBLIC ${DBUS_INCLUDE_DIRS})
target_link_libraries(npud_core PRIVATE nnfw_lib_misc)
target_link_libraries(npud_core PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_core PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud_core PRIVATE dl)
target_link_libraries(npud_core PRIVATE npud_dbus)

if(ENVVAR_NPUD_CONFIG)
  target_compile_definitions(npud_core PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_NPUD_CONFIG)

# npud executable
add_executable(npud ${MAIN_SOURCE_FILE})

set_target_properties(npud PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(npud PRIVATE npud_core)

install(TARGETS npud DESTINATION ${CMAKE_INSTALL_BINDIR})
