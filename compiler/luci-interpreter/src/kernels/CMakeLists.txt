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
if (NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(${LUCI_INTERPRETER_KERNELS} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)
target_include_directories(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})

target_link_libraries(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_CORE})
target_link_libraries(${LUCI_INTERPRETER_KERNELS} PRIVATE nncc_common)

add_pal_to_target(${LUCI_INTERPRETER_KERNELS})

# Disable array-bounds error from tensorflow lite header.
# "target_include_directories(luci_interpreter_linux_pal SYSTEM PRIVATE ${TensorFlowSource_DIR})"
# in add_pal_to_target does not resolve array-bounds error.
# TODO Find way to remove this option in future.
target_compile_options(${LUCI_INTERPRETER_KERNELS} PRIVATE -Wno-error=array-bounds)

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
