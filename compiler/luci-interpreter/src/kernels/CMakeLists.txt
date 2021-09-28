set(SOURCES
        BinaryOpCommon.h
        Utils.h
        Utils.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/TestMemoryManager.h"
        ${LUCI_INTERPRETER_SOURCE_DIR}/TestMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/SimpleMemoryManager.h"
        ${LUCI_INTERPRETER_SOURCE_DIR}/SimpleMemoryManager.cpp)

macro(REGISTER_KERNEL NODE)
  list(APPEND SOURCES "${NODE}.h")
  list(APPEND SOURCES "${NODE}.cpp")
endmacro(REGISTER_KERNEL)

include(${KERNEL_REGISTER_FILE})

add_library(${LUCI_INTERPRETER_KERNELS} STATIC ${SOURCES})
set_target_properties(${LUCI_INTERPRETER_KERNELS} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})

target_link_libraries(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_CORE})
target_link_libraries(${LUCI_INTERPRETER_KERNELS} PRIVATE nncc_common)

add_pal_to_target(${LUCI_INTERPRETER_KERNELS})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

macro(REGISTER_KERNEL NODE)
  list(APPEND TEST_SOURCES "${NODE}.test.cpp")
endmacro(REGISTER_KERNEL)

include(${KERNEL_REGISTER_FILE})

list(APPEND TEST_SOURCES TestUtils.h TestUtils.cpp)

GTest_AddTest(${LUCI_INTERPRETER_KERNELS}_test ${TEST_SOURCES})
target_link_libraries(${LUCI_INTERPRETER_KERNELS}_test ${LUCI_INTERPRETER_KERNELS})
