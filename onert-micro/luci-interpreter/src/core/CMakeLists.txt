set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/DataType.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/Tensor.h"
    Kernel.h
    KernelParams.h
    RuntimeGraph.h
    RuntimeGraph.cpp
    RuntimeModule.h
    Tensor.cpp)

add_library(${LUCI_INTERPRETER_CORE} STATIC ${SOURCES})
if(NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(${LUCI_INTERPRETER_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)

add_subdirectory(reader)

target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC "luci_micro_circle_reader${READER_SUFFIX}")
target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC luci_micro_circle_schema)

target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
