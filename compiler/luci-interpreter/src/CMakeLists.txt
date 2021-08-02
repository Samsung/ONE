include(${LUCI_INTERPRETER_PAL_DIR}/pal.cmake)

initialize_pal()

if (NOT PAL_INITIALIZED)
  return()
endif()

message(STATUS "LUCI INTERPRETER BEGIN")

add_subdirectory(core)
message(STATUS "LUCI INTERPRETER CORE")
add_subdirectory(kernels)
message(STATUS "LUCI INTERPRETER KERNELS")
add_subdirectory(loader)
message(STATUS "LUCI INTERPRETER LOADER")

message(STATUS "LUCI INTERPTER INITALIZED")

set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/Interpreter.h"
    Interpreter.cpp)

add_library(luci_interpreter SHARED ${SOURCES})
target_include_directories(luci_interpreter PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(luci_interpreter PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(luci_interpreter
    PUBLIC luci_lang luci_interpreter_loader luci_interpreter_core
    PRIVATE nncc_common)

install(TARGETS luci_interpreter DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
