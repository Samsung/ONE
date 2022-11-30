nnfw_find_package(GLib2.0 REQUIRED)
nnfw_find_package(TRIXEngine QUIET 2.5.0)

if(NOT TRIXEngine_FOUND)
  return()
endif(NOT TRIXEngine_FOUND)

file(GLOB_RECURSE TESTS "*.cc")

add_executable(npud_test_client ${TESTS})

set_target_properties(npud_test_client PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_test_client PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_include_directories(npud_test_client PUBLIC ${DBUS_INCLUDE_DIRS})
target_link_libraries(npud_test_client PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_test_client PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud_test_client PRIVATE npud_dbus)
target_link_libraries(npud_test_client PRIVATE trix_engine)

install(TARGETS npud_test_client DESTINATION bin)
