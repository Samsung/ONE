if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

set(NPUD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

include(FindPkgConfig)
pkg_check_modules(GIO REQUIRED "gio-2.0 gio-unix-2.0")

find_program(GDBUS_CODEGEN NAMES gdbus-codegen)
if (NOT GDBUS_CODEGEN)
    message(SEND_ERROR "Could not find gdbus-codegen")
endif(NOT GDBUS_CODEGEN)

set(DBUS_CORE "dbus-core")
set(DBUS_SOURCE_FILE "${DBUS_CORE}.c")
set(DBUS_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
set(DBUS_CORE_INTROSPECTION_XML "org.tizen.npud.xml")
set(DBUS_SERVICE_CONFIG "org.tizen.npud.conf")

add_custom_command(OUTPUT ${DBUS_SOURCE_FILE}
  COMMAND ${GDBUS_CODEGEN} --interface-prefix org.tizen --generate-c-code=${DBUS_CORE} ${CMAKE_CURRENT_SOURCE_DIR}/${DBUS_CORE_INTROSPECTION_XML}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${DBUS_CORE_INTROSPECTION_XML})

add_library(npud_dbus SHARED ${DBUS_SOURCE_FILE})

target_compile_options(npud_dbus PUBLIC ${GIO_CFLAGS})
target_link_libraries(npud_dbus PRIVATE ${GIO_LDFLAGS})

install(TARGETS npud_dbus LIBRARY DESTINATION lib)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET npud_dbus POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:npud_dbus>)
endif()

add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(tests)
