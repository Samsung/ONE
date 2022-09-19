if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

nnfw_find_package(GLib2.0 REQUIRED)

include(FindPkgConfig)
pkg_check_modules(GIO REQUIRED "gio-2.0 gio-unix-2.0")

find_program(GDBUS_CODEGEN NAMES gdbus-codegen)
if (NOT GDBUS_CODEGEN)
    message(SEND_ERROR "Could not find gdbus-codegen")
endif(NOT GDBUS_CODEGEN)

set(DBUS_CORE "dbus-core")
set(DBUS_SOURCE_FILE "${DBUS_CORE}.c")
set(DBUS_CORE_INTROSPECTION_XML "core/org.tizen.npud.xml")
set(DBUS_SERVICE_CONFIG "org.tizen.npud.conf")

add_custom_command(OUTPUT ${DBUS_SOURCE_FILE}
  COMMAND ${GDBUS_CODEGEN} --interface-prefix org.tizen --generate-c-code=${DBUS_CORE} ${CMAKE_CURRENT_SOURCE_DIR}/${DBUS_CORE_INTROSPECTION_XML}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${DBUS_CORE_INTROSPECTION_XML})

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE TESTS "tests/*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_executable(npud ${SOURCES} ${DBUS_SOURCE_FILE})
set_target_properties(npud PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(npud PUBLIC ${GIO_CFLAGS})
target_include_directories(npud PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(npud PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud PRIVATE nnfw_lib_misc)
target_link_libraries(npud PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud PRIVATE ${GIO_LDFLAGS})

if(ENVVAR_NPUD_CONFIG)
  target_compile_definitions(npud PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_NPUD_CONFIG)

install(TARGETS npud DESTINATION bin)
install(FILES ${DBUS_SERVICE_CONFIG} DESTINATION etc/dbus-1/systemd.d)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

list(FILTER SOURCES EXCLUDE REGEX ".*main.cc$")

add_executable(npud_test ${TESTS} ${SOURCES})
set_target_properties(npud_test PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud_test PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud_test PRIVATE nnfw_lib_misc)
target_link_libraries(npud_test PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_test PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud_test PRIVATE gtest_main dl)

install(TARGETS npud_test DESTINATION unittest_standalone)
