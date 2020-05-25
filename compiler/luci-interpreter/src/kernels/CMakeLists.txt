nnas_find_package(TensorFlowSource EXACT 2.1.0 QUIET)
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.1.0 QUIET)
nnas_find_package(TensorFlowEigenSource EXACT 2.1.0 QUIET)
find_package(Threads REQUIRED)
nnas_find_package(GTest REQUIRED)

if (NOT TensorFlowSource_FOUND OR
    NOT TensorFlowGEMMLowpSource_FOUND OR
    NOT TensorFlowEigenSource_FOUND)
  return()
endif ()

set(SOURCES
    Add.h
    Add.cpp
    AveragePool2D.h
    AveragePool2D.cpp
    Concatenation.h
    Concatenation.cpp
    Conv2D.h
    Conv2D.cpp
    DepthwiseConv2D.h
    DepthwiseConv2D.cpp
    FullyConnected.h
    FullyConnected.cpp
    MaxPool2D.h
    MaxPool2D.cpp
    Mul.h
    Mul.cpp
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
    PRIVATE nncc_common Threads::Threads)


set(TEST_SOURCES
    TestUtils.h
    TestUtils.cpp
    Add.test.cpp
    AveragePool2D.test.cpp
    Concatenation.test.cpp
    Conv2D.test.cpp
    DepthwiseConv2D.test.cpp
    FullyConnected.test.cpp
    MaxPool2D.test.cpp
    Mul.test.cpp
    Reshape.test.cpp
    Softmax.test.cpp)

GTest_AddTest(luci_interpreter_kernels_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_kernels_test luci_interpreter_kernels)
