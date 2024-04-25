nnas_find_package(FlatBuffers EXACT 23.5.26 REQUIRED)

if (NOT FlatBuffers_FOUND)
    return()
endif ()

FlatBuffers_Target(mir_tflite_schema
        OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/schema"
        SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schema"
        SCHEMA_FILES schema.fbs)


set(MIR_TFLITE_IMPORTER_SOURCES
        tflite_importer.cpp
        tflite_op_creator.cpp
        tflite_op_creator.h)

add_library(mir_tflite_importer STATIC ${MIR_TFLITE_IMPORTER_SOURCES})
set_target_properties(mir_tflite_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mir_tflite_importer PUBLIC ../../include/mir_tflite_importer)
target_link_libraries(mir_tflite_importer PUBLIC mir PRIVATE mir_tflite_schema nncc_common)
