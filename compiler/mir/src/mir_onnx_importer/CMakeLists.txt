nnas_find_package(ONNXSource EXACT 1.6.0 QUIET)
nnas_find_package(Protobuf QUIET)

if (NOT ONNXSource_FOUND)
    return()
endif ()

if (NOT Protobuf_FOUND)
    return()
endif ()

if (ENABLE_EXCLUDE_ME)
    if (NOT TARGET mir_interpreter)
        return()
    endif ()
endif ()

Protobuf_Generate(MIR_ONNX_PROTO
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${ONNXSource_DIR}
        onnx/onnx.proto)

add_library(mir_onnx_proto STATIC ${MIR_ONNX_PROTO_SOURCES})
set_target_properties(mir_onnx_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mir_onnx_proto PUBLIC ${MIR_ONNX_PROTO_INCLUDE_DIRS})
target_link_libraries(mir_onnx_proto PUBLIC libprotobuf)

set(MIR_ONNX_IMPORTER_SOURCES
        AttributeHelpers.h
        ConvPoolHelpers.cpp
        ConvPoolHelpers.h
        ONNXHelpers.cpp
        ONNXHelpers.h
        ONNXImporterImpl.cpp
        ONNXNodeConverterRegistry.h
        ONNXNodeConverterRegistry.cpp
        ONNXOpRegistration.h
        Op/Abs.cpp
        Op/Abs.h
        Op/Add.cpp
        Op/Add.h
        Op/AveragePool.cpp
        Op/AveragePool.h
        Op/BatchNormalization.cpp
        Op/BatchNormalization.h
        Op/Concat.cpp
        Op/Concat.h
        Op/Constant.cpp
        Op/Constant.h
        Op/Conv.cpp
        Op/Conv.h
        Op/ConvTranspose.cpp
        Op/ConvTranspose.h
        Op/Div.cpp
        Op/Div.h
        Op/Dropout.cpp
        Op/Dropout.h
        Op/Equal.cpp
        Op/Equal.h
        Op/Expand.cpp
        Op/Expand.h
        Op/Flatten.cpp
        Op/Flatten.h
        Op/Gather.cpp
        Op/Gather.h
        Op/Greater.cpp
        Op/Greater.h
        Op/Gemm.cpp
        Op/Gemm.h
        Op/Identity.cpp
        Op/Identity.h
        Op/Less.cpp
        Op/Less.h
        Op/MatMul.cpp
        Op/MatMul.h
        Op/GlobalAveragePool.cpp
        Op/GlobalAveragePool.h
        Op/Max.cpp
        Op/Max.h
        Op/MaxPool.cpp
        Op/MaxPool.h
        Op/Mul.cpp
        Op/Mul.h
        Op/Pad.cpp
        Op/Pad.h
        Op/Reciprocal.cpp
        Op/Reciprocal.h
        Op/ReduceMean.cpp
        Op/ReduceMean.h
        Op/Relu.cpp
        Op/Relu.h
        Op/Reshape.cpp
        Op/Reshape.h
        Op/Shape.cpp
        Op/Shape.h
        Op/Sigmoid.cpp
        Op/Sigmoid.h
        Op/Softmax.cpp
        Op/Softmax.h
        Op/Sqrt.cpp
        Op/Sqrt.h
        Op/Sub.cpp
        Op/Sub.h
        Op/Sum.cpp
        Op/Sum.h
        Op/Tanh.cpp
        Op/Tanh.h
        Op/Transpose.cpp
        Op/Transpose.h
        Op/Unsqueeze.cpp
        Op/Unsqueeze.h
        Op/Upsample.cpp
        Op/Upsample.h)

add_library(mir_onnx_importer STATIC ${MIR_ONNX_IMPORTER_SOURCES})
set_target_properties(mir_onnx_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mir_onnx_importer PUBLIC ../../include/mir_onnx_importer)
target_include_directories(mir_onnx_importer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mir_onnx_importer PUBLIC mir mir_onnx_proto PRIVATE mir_interpreter nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TEST_SOURCES "*.test.cpp")
GTest_AddTest(mir_onnx_importer_test ${TEST_SOURCES})
target_link_libraries(mir_onnx_importer_test mir_onnx_importer)
