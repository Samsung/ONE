nnas_find_package(CaffeProto QUIET)

if (NOT CaffeProto_FOUND)
    return()
endif ()

set(MIR_CAFFE_IMPORTER_SOURCES
        caffe_importer.cpp
        caffe_op_creator.cpp
        caffe_op_creator.h
        caffe_op_types.h)

add_library(mir_caffe_importer STATIC ${MIR_CAFFE_IMPORTER_SOURCES})
set_target_properties(mir_caffe_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mir_caffe_importer PUBLIC ../../include/mir_caffe_importer)
target_link_libraries(mir_caffe_importer PUBLIC mir PRIVATE caffeproto)
# to prevent _GLIBCXX17_DEPRECATED warning as error
# target_link_libraries(mir_caffe_importer PRIVATE nncc_common)
