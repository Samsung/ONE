nnfw_find_package(GLib2.0 REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE MAIN_SOURCE_FILE "main.cc")
list(REMOVE_ITEM SOURCES ${MAIN_SOURCE_FILE})

add_library(npud_core SHARED ${SOURCES})

set_target_properties(npud_core PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud_core PUBLIC ${DBUS_INCLUDE_DIRS})
target_include_directories(npud_core PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud_core PRIVATE nnfw_lib_misc)
target_link_libraries(npud_core PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_core PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud_core PRIVATE npud_dbus)

if(ENVVAR_NPUD_CONFIG)
  target_compile_definitions(npud_core PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_NPUD_CONFIG)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET npud_core POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:npud_core>)
endif()

install(TARGETS npud_core LIBRARY DESTINATION lib)

# npud executable
add_executable(npud ${MAIN_SOURCE_FILE})

target_link_libraries(npud PRIVATE npud_core)

install(TARGETS npud DESTINATION bin)
