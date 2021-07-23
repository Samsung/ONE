if (NOT LUCI_MICRO)
  find_package(Threads REQUIRED)
endif ()
set(KERNEL_REGISTER_FILE "${LUCI_INTERPRETER_SOURCE_DIR}/KernelsToBuild.lst")
set(APPEND SOURCES
        BinaryOpCommon.h
        Utils.h
        Utils.cpp
        ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc)

macro(REGISTER_KERNEL NODE)
  list(APPEND SOURCES "${NODE}.h")
  list(APPEND SOURCES "${NODE}.cpp")
endmacro(REGISTER_KERNEL)

if(EXISTS "${KERNEL_REGISTER_FILE}")
  include(${KERNEL_REGISTER_FILE})
endif(EXISTS "${KERNEL_REGISTER_FILE}")

add_library(luci_interpreter_kernels STATIC ${SOURCES})
set_target_properties(luci_interpreter_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luci_interpreter_kernels PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})
target_include_directories(luci_interpreter_kernels SYSTEM PRIVATE
        "${TensorFlowRuySource_DIR}"
        "${TensorFlowGEMMLowpSource_DIR}"
        "${TensorFlowEigenSource_DIR}"
        "${TensorFlowSource_DIR}")

if (NOT LUCI_MICRO)
  target_link_libraries(luci_interpreter_kernels
          PUBLIC luci_interpreter_core
          PRIVATE nncc_common Threads::Threads)
else ()
  target_link_libraries(luci_interpreter_kernels
          PUBLIC luci_interpreter_core)
endif ()

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

macro(REGISTER_KERNEL NODE)
  list(APPEND TEST_SOURCES "${NODE}.test.cpp")
endmacro(REGISTER_KERNEL)

if(EXISTS "${KERNEL_REGISTER_FILE}")
  include(${KERNEL_REGISTER_FILE})
endif(EXISTS "${KERNEL_REGISTER_FILE}")

list(APPEND TEST_SOURCES TestUtils.h TestUtils.cpp)

GTest_AddTest(luci_interpreter_kernels_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_kernels_test luci_interpreter_kernels)
