add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(loader)

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
