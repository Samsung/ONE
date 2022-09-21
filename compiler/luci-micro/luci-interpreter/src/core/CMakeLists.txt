set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/DataType.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/Tensor.h"
    Kernel.h
    KernelParams.h
    RuntimeGraph.h
    RuntimeGraph.cpp
    RuntimeModule.h
    Tensor.cpp)

set(READER_SOURCE
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/CircleMicroReader.h"
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/CircleMicroReaderHelper.h"
        CircleMicroReader.cpp
        CircleMicroReaderHelper.cpp
        )

add_library(circle_reader STATIC ${READER_SOURCE})
target_link_libraries(circle_reader PUBLIC circle_schema)
target_include_directories(circle_reader PUBLIC "${GENERATED_INCLUDE_DIR}")
target_include_directories(circle_reader PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")

add_library(${LUCI_INTERPRETER_CORE} STATIC ${SOURCES})
if (NOT NNCC_LIBRARY_NO_PIC)
    set_target_properties(${LUCI_INTERPRETER_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)

target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC circle_reader)

target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
