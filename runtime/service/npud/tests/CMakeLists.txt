if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnfw_find_package(GLib2.0 REQUIRED)

# For dbus test
include(FindPkgConfig)
pkg_check_modules(GIO REQUIRED "gio-2.0 gio-unix-2.0")

file(GLOB_RECURSE TESTS "*.cc")

add_executable(npud_test ${TESTS})
set_target_properties(npud_test PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(npud_test PUBLIC ${GIO_CFLAGS})
target_include_directories(npud_test PUBLIC ${NPUD_INCLUDE_DIRS})
target_include_directories(npud_test PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud_test PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_test PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud_test PRIVATE ${GIO_LDFLAGS})
target_link_libraries(npud_test PRIVATE npud_core)
target_link_libraries(npud_test PRIVATE npud_dbus)
target_link_libraries(npud_test PRIVATE gtest_main dl)

install(TARGETS npud_test DESTINATION unittest_standalone)
