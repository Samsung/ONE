set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/DataType.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/Tensor.h"
    EventNotifier.h
    Kernel.h
    KernelParams.h
    RuntimeGraph.h
    RuntimeGraph.cpp
    RuntimeModule.h
    Tensor.cpp)

add_library(${LUCI_INTERPRETER_CORE} STATIC ${SOURCES})
set_target_properties(${LUCI_INTERPRETER_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC luci_lang)
target_link_libraries(${LUCI_INTERPRETER_CORE} PRIVATE nncc_common)
