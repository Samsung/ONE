find_package(HDF5 COMPONENTS CXX QUIET)
if(NOT HDF5_FOUND)
  message(WARNING "HDF5 not found, tflite_model_generator not available")
  return()
endif(NOT HDF5_FOUND)

nnas_find_package(FlatBuffers QUIET)
if(NOT FlatBuffers_FOUND)
  message(WARNING "FlatBuffers not found, tflite_model_generator not available")
  return()
endif(NOT FlatBuffers_FOUND)

if (NOT TARGET tflite_schema)
  message(WARNING "tflite_schema not found, tflite_model_generator not available")
  return()
endif (NOT TARGET tflite_schema)

set(TFLITE_MODEL_GENERATOR_SOURCES "ModelGenerator.cpp" "TFLiteRandomModelBuilder.cpp" "Tree.cpp")

add_executable(tflite_model_generator ${TFLITE_MODEL_GENERATOR_SOURCES})

target_link_libraries(tflite_model_generator PRIVATE flatbuffers)
target_link_libraries(tflite_model_generator PRIVATE tflite_schema)
target_link_libraries(tflite_model_generator PRIVATE ${HDF5_CXX_LIBRARIES})

target_include_directories(tflite_model_generator PRIVATE ${HDF5_INCLUDE_DIRS})
