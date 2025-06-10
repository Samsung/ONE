set(MIR_SOURCES
        src/ops/AvgPool2DOp.cpp
        src/ops/BinaryElementwiseOp.cpp
        src/ops/BroadcastOp.cpp
        src/ops/ConcatOp.cpp
        src/ops/Conv2DOp.cpp
        src/ops/DeConv2DOp.cpp
        src/ops/DepthwiseConv2DOp.cpp
        src/ops/FullyConnectedOp.cpp
        src/ops/GatherOp.cpp
        src/ops/MaxPool2DOp.cpp
        src/ops/PadOp.cpp
        src/ops/ReduceOp.cpp
        src/ops/SqueezeOp.cpp
        src/ops/SliceOp.cpp
        src/ops/TransposeOp.cpp
        src/DotGraph.cpp
        src/DotNodeBuilder.cpp
        src/Graph.cpp
        src/GraphPatternMatcher.cpp
        src/Index.cpp
        src/IrDotDumper.cpp
        src/Operation.cpp
        src/Shape.cpp
        src/Tensor.cpp
        src/TensorVariant.cpp
        src/Visitor.cpp)

add_library(mir STATIC ${MIR_SOURCES})
target_include_directories(mir PUBLIC include)
target_link_libraries(mir PUBLIC adtidas)
# to prevent _GLIBCXX17_DEPRECATED warning as error
# target_link_libraries(mir PRIVATE nncc_common)
target_link_libraries(mir PUBLIC nncc_coverage)

set_target_properties(mir PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(mir PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(unittests)

add_subdirectory(src/mir_caffe_importer)
add_subdirectory(src/mir_caffe2_importer)
add_subdirectory(src/mir_tflite_importer)
add_subdirectory(src/mir_onnx_importer)
