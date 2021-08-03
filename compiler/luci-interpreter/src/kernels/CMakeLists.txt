set(KERNEL_REGISTER_FILE ${LUCI_INTERPRETER_PAL_DIR}/KernelsToBuild.lst)

set(SOURCES
        BinaryOpCommon.h
        Utils.h
        Utils.cpp
        ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc
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
target_include_directories(${LUCI_INTERPRETER_KERNELS} SYSTEM PRIVATE
        "${TensorFlowRuySource_DIR}"
        "${TensorFlowGEMMLowpSource_DIR}"
        "${TensorFlowEigenSource_DIR}"
        "${TensorFlowSource_DIR}")

if (NOT LUCI_MICRO)
  target_link_libraries(${LUCI_INTERPRETER_KERNELS}
          PUBLIC ${LUCI_INTERPRETER_CORE}
          PRIVATE nncc_common Threads::Threads)
else ()
  target_link_libraries(${LUCI_INTERPRETER_KERNELS}
          PUBLIC ${LUCI_INTERPRETER_CORE})
endif()

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
