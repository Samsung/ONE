if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

set(NPUD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

nnfw_find_package(Gio2.0 REQUIRED)
nnfw_find_package(Giounix2.0 REQUIRED)

find_program(GDBUS_CODEGEN NAMES gdbus-codegen)
if (NOT GDBUS_CODEGEN)
    message(SEND_ERROR "Could not find gdbus-codegen")
endif(NOT GDBUS_CODEGEN)

set(DBUS_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
set(DBUS_INTERFACE "org.tizen.npud")
set(DBUS_NAMESPACE "Npud")
set(DBUS_INTROSPECTION_XML "org.tizen.npud.xml")
set(DBUS_CORE "dbus-core")
set(DBUS_CORE_SOURCE "${DBUS_CORE}.c")
set(DBUS_CONFIG_FILE "org.tizen.npud.conf")

add_custom_command(OUTPUT ${DBUS_CORE_SOURCE}
  COMMAND ${GDBUS_CODEGEN}
  --generate-c-code ${DBUS_CORE}
  --interface-prefix ${DBUS_INTERFACE}
  --c-namespace ${DBUS_NAMESPACE}
  ${CMAKE_CURRENT_SOURCE_DIR}/${DBUS_INTROSPECTION_XML}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${DBUS_INTROSPECTION_XML})

add_library(npud_dbus STATIC ${DBUS_CORE_SOURCE})

target_include_directories(npud_dbus PUBLIC ${GIO2.0_INCLUDE_DIRS})
target_include_directories(npud_dbus PUBLIC ${GIO_UNIX_2.0_INCLUDE_DIRS})
target_link_libraries(npud_dbus PRIVATE ${GIO2.0_LIBRARIES})
target_link_libraries(npud_dbus PRIVATE ${GIO_UNIX_2.0_LIBRARIES})

install(FILES ${DBUS_CONFIG_FILE} DESTINATION share)

add_subdirectory(core)
add_subdirectory(tests)
add_subdirectory(backend)
