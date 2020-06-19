add_subdirectory(core)
add_subdirectory(kernels)

set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/Interpreter.h"
    Interpreter.cpp
    KernelBuilder.h
    KernelBuilder.cpp
    KernelMap.h
    RuntimeToIR.h
    TensorMap.h)

add_library(luci_interpreter SHARED ${SOURCES})
target_include_directories(luci_interpreter PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(luci_interpreter PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(luci_interpreter PUBLIC luci_lang)
target_link_libraries(luci_interpreter PUBLIC luci_interpreter_core)
target_link_libraries(luci_interpreter PRIVATE luci_interpreter_kernels)
target_link_libraries(luci_interpreter PRIVATE nncc_common)

install(TARGETS luci_interpreter DESTINATION lib)
