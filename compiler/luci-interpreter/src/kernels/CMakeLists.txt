nnas_find_package(TensorFlowSource EXACT 2.1.0 QUIET)
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.1.0 QUIET)
nnas_find_package(TensorFlowEigenSource EXACT 2.1.0 QUIET)
nnas_find_package(GTest REQUIRED)

if (NOT TensorFlowSource_FOUND OR
    NOT TensorFlowGEMMLowpSource_FOUND OR
    NOT TensorFlowEigenSource_FOUND)
  return()
endif ()

set(SOURCES
    FullyConnected.h
    FullyConnected.cpp
    MaxPool2D.cpp
    Reshape.h
    Reshape.cpp
    Softmax.h
    Softmax.cpp
    Utils.h
    Utils.cpp)

add_library(luci_interpreter_kernels STATIC ${SOURCES})
set_target_properties(luci_interpreter_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luci_interpreter_kernels PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})
target_include_directories(luci_interpreter_kernels SYSTEM PRIVATE
    "${TensorFlowGEMMLowpSource_DIR}"
    "${TensorFlowEigenSource_DIR}"
    "${TensorFlowSource_DIR}")
target_link_libraries(luci_interpreter_kernels
    PUBLIC luci_interpreter_core
    PRIVATE nncc_common)


set(TEST_SOURCES
    TestUtils.h
    TestUtils.cpp
    FullyConnected.test.cpp
    MaxPool2D.test.cpp
    Reshape.test.cpp
    Softmax.test.cpp)

GTest_AddTest(luci_interpreter_kernels_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_kernels_test luci_interpreter_kernels)
