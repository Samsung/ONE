find_package(Threads REQUIRED)

set(SOURCES
    Add.h
    Add.cpp
    ArgMax.h
    ArgMax.cpp
    AveragePool2D.h
    AveragePool2D.cpp
    Concatenation.h
    Concatenation.cpp
    Conv2D.h
    Conv2D.cpp
    DepthToSpace.h
    DepthToSpace.cpp
    DepthwiseConv2D.h
    DepthwiseConv2D.cpp
    Div.h
    Div.cpp
    Elu.h
    Elu.cpp
    Exp.h
    Exp.cpp
    Floor.h
    Floor.cpp
    FloorDiv.h
    FloorDiv.cpp
    Equal.h
    Equal.cpp
    FullyConnected.h
    FullyConnected.cpp
    Greater.h
    Greater.cpp
    GreaterEqual.h
    GreaterEqual.cpp
    If.h
    If.cpp
    InstanceNorm.h
    InstanceNorm.cpp
    L2Normalize.h
    L2Normalize.cpp
    L2Pool2D.h
    L2Pool2D.cpp
    LeakyRelu.h
    LeakyRelu.cpp
    Less.h
    Less.cpp
    LessEqual.h
    LessEqual.cpp
    LocalResponseNormalization.h
    LocalResponseNormalization.cpp
    LogicalAnd.h
    LogicalAnd.cpp
    LogicalNot.h
    LogicalNot.cpp
    LogicalOr.h
    LogicalOr.cpp
    Logistic.h
    Logistic.cpp
    LogSoftmax.h
    LogSoftmax.cpp
    Maximum.h
    Maximum.cpp
    MaxPool2D.h
    MaxPool2D.cpp
    Mean.h
    Mean.cpp
    Minimum.h
    Minimum.cpp
    Mul.h
    Mul.cpp
    Neg.h
    Neg.cpp
    NotEqual.h
    NotEqual.cpp
    Pack.h
    Pack.cpp
    Pad.h
    Pad.cpp
    Pow.h
    Pow.cpp
    Prelu.h
    Prelu.cpp
    Relu.h
    Relu.cpp
    Relu6.h
    Relu6.cpp
    Reshape.h
    Reshape.cpp
    ResizeBilinear.h
    ResizeBilinear.cpp
    ResizeNearestNeighbor.h
    ResizeNearestNeighbor.cpp
    Reverse.h
    Reverse.cpp
    Rsqrt.h
    Rsqrt.cpp
    Slice.h
    Slice.cpp
    Softmax.h
    Softmax.cpp
    SpaceToDepth.h
    SpaceToDepth.cpp
    Split.h
    Split.cpp
    StridedSlice.h
    StridedSlice.cpp
    Sqrt.h
    Sqrt.cpp
    SquaredDifference.h
    SquaredDifference.cpp
    Squeeze.h
    Squeeze.cpp
    Sub.h
    Sub.cpp
    Tanh.h
    Tanh.cpp
    Transpose.h
    Transpose.cpp
    TransposeConv.h
    TransposeConv.cpp
    Unpack.h
    Unpack.cpp)

list(APPEND SOURCES
    BinaryOpCommon.h
    Utils.h
    Utils.cpp
    ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc)

add_library(luci_interpreter_kernels STATIC ${SOURCES})
set_target_properties(luci_interpreter_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luci_interpreter_kernels PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})
target_include_directories(luci_interpreter_kernels SYSTEM PRIVATE
    "${TensorFlowRuySource_DIR}"
    "${TensorFlowGEMMLowpSource_DIR}"
    "${TensorFlowEigenSource_DIR}"
    "${TensorFlowSource_DIR}")
target_link_libraries(luci_interpreter_kernels
    PUBLIC luci_interpreter_core
    PRIVATE nncc_common Threads::Threads)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCES
    Add.test.cpp
    ArgMax.test.cpp
    AveragePool2D.test.cpp
    Concatenation.test.cpp
    Conv2D.test.cpp
    DepthToSpace.test.cpp
    DepthwiseConv2D.test.cpp
    Div.test.cpp
    Elu.test.cpp
    Exp.test.cpp
    Floor.test.cpp
    FloorDiv.test.cpp
    Equal.test.cpp
    FullyConnected.test.cpp
    Greater.test.cpp
    GreaterEqual.test.cpp
    If.test.cpp
    InstanceNorm.test.cpp
    L2Normalize.test.cpp
    L2Pool2D.test.cpp
    LeakyRelu.test.cpp
    Less.test.cpp
    LessEqual.test.cpp
    LocalResponseNormalization.test.cpp
    LogicalAnd.test.cpp
    LogicalNot.test.cpp
    LogicalOr.test.cpp
    Logistic.test.cpp
    LogSoftmax.test.cpp
    Maximum.test.cpp
    MaxPool2D.test.cpp
    Mean.test.cpp
    Minimum.test.cpp
    Mul.test.cpp
    Neg.test.cpp
    NotEqual.test.cpp
    Pack.test.cpp
    Pad.test.cpp
    Pow.test.cpp
    Prelu.test.cpp
    Relu.test.cpp
    Relu6.test.cpp
    Reshape.test.cpp
    ResizeBilinear.test.cpp
    ResizeNearestNeighbor.test.cpp
    Reverse.test.cpp
    Rsqrt.test.cpp
    Slice.test.cpp
    Softmax.test.cpp
    SpaceToDepth.test.cpp
    Split.test.cpp
    StridedSlice.test.cpp
    Sqrt.test.cpp
    SquaredDifference.test.cpp
    Squeeze.test.cpp
    Sub.test.cpp
    Tanh.test.cpp
    Transpose.test.cpp
    TransposeConv.test.cpp
    Unpack.test.cpp)

list(APPEND TEST_SOURCES TestUtils.h TestUtils.cpp)

GTest_AddTest(luci_interpreter_kernels_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_kernels_test luci_interpreter_kernels)
