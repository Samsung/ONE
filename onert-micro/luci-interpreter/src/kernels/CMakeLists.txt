set(SOURCES
        BinaryOpCommon.h
        Utils.h
        Utils.cpp
        Builders.h
        KernelBuilder.h
        KernelBuilder.cpp
        SISOKernel.h
        TISOKernel.h
        MISOKernel.h
        PadCommon.cpp)

macro(REGISTER_KERNEL OPERATOR, NODE)
  list(APPEND SOURCES "${NODE}.cpp")
endmacro(REGISTER_KERNEL)

macro(REGISTER_TRAIN_KERNEL OPERATOR, NODE)
  list(APPEND SOURCES "${NODE}.train.cpp")
endmacro(REGISTER_TRAIN_KERNEL)

include(${KERNEL_REGISTER_FILE})

include("${LUCI_INTERPRETER_PAL_DIR}/KernelsToTrain.lst")

add_library(${LUCI_INTERPRETER_KERNELS} STATIC ${SOURCES})
if (NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(${LUCI_INTERPRETER_KERNELS} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)
target_include_directories(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})

target_link_libraries(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_CORE})

target_include_directories(${LUCI_INTERPRETER_KERNELS} PUBLIC ${LUCI_INTERPRETER_PAL_COMMON_DIR})
add_pal_to_target(${LUCI_INTERPRETER_KERNELS})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

macro(REGISTER_KERNEL OPERATOR, NODE)
  list(APPEND TEST_SOURCES "${NODE}.test.cpp")
endmacro(REGISTER_KERNEL)

include(${KERNEL_REGISTER_FILE})

list(APPEND TEST_SOURCES TestUtils.h TestUtils.cpp)

GTest_AddTest(${LUCI_INTERPRETER_KERNELS}_test ${TEST_SOURCES})
target_link_libraries(${LUCI_INTERPRETER_KERNELS}_test onert_micro_coverage)
target_link_libraries(${LUCI_INTERPRETER_KERNELS} PUBLIC onert_micro_coverage)
target_link_libraries(${LUCI_INTERPRETER_KERNELS}_test ${LUCI_INTERPRETER_LOADER})
