nnas_find_package(PytorchSource QUIET)
nnas_find_package(Protobuf QUIET)

if (NOT PytorchSource_FOUND OR NOT Protobuf_FOUND)
    return()
endif()

Protobuf_Generate(CAFFE2_PROTO "${CMAKE_CURRENT_BINARY_DIR}/generated/caffe2"
        "${PytorchSource_DIR}" "caffe2/proto/caffe2.proto")

add_library(caffe2proto STATIC ${CAFFE2_PROTO_SOURCES})
set_target_properties(caffe2proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(caffe2proto PUBLIC ${CAFFE2_PROTO_INCLUDE_DIRS})
target_link_libraries(caffe2proto PUBLIC libprotobuf)


set(MIR_CAFFE2_IMPORTER_SOURCES
        caffe2_importer.cpp
        caffe2_op_creator.cpp
        caffe2_op_creator.h
        caffe2_op_types.h
        caffe2_proto_helper.cpp
        caffe2_proto_helper.h)

add_library(mir_caffe2_importer STATIC ${MIR_CAFFE2_IMPORTER_SOURCES})
set_target_properties(mir_caffe2_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mir_caffe2_importer PUBLIC ../../include/mir_caffe2_importer)
target_link_libraries(mir_caffe2_importer PUBLIC mir PRIVATE caffe2proto)
# to prevent _GLIBCXX17_DEPRECATED warning as error
# target_link_libraries(mir_caffe2_importer PRIVATE nncc_common)
