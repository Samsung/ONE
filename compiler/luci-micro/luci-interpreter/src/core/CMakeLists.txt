set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/DataType.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/Tensor.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/CircleMicroReader.h"
    Kernel.h
    KernelParams.h
    RuntimeGraph.h
    RuntimeGraph.cpp
    RuntimeModule.h
    Tensor.cpp
    CircleMicroReader.cpp)

add_library(${LUCI_INTERPRETER_CORE} STATIC ${SOURCES})
if (NOT NNCC_LIBRARY_NO_PIC)
    set_target_properties(${LUCI_INTERPRETER_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)

target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC mio_circle04)
target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC mio_circle04_helper)

target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC luci_lang)
target_link_libraries(${LUCI_INTERPRETER_CORE} PRIVATE nncc_common)
