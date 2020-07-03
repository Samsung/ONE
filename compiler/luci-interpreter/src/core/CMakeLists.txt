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

add_library(luci_interpreter_core STATIC ${SOURCES})
set_target_properties(luci_interpreter_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luci_interpreter_core PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(luci_interpreter_core PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(luci_interpreter_core PUBLIC luci_lang)
target_link_libraries(luci_interpreter_core PRIVATE nncc_common)
