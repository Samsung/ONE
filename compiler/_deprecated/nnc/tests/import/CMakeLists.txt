# TODO: This CMakeLists.txt declares two targets that build executables
# that can be used to run following system tests:
#
# Caffe Mobilenet model is imported successfully
# Caffe Inception-v3 model is imported successfully
# Tensorflow Lite Mobilenet model is imported successfully
# Tensorflow Lite Inception-v3 model is imported successfully
#
# These executables are not executed anywhere because it is not yet decided
# how to store large files (in this case, files with models), and how to run all system tests.
# As soon as it is decided, model files should be added, as well as the code that runs the tests.
if(NNC_FRONTEND_TFLITE_ENABLED)
  add_executable(system_test_import_tflite tflite.cpp)
  target_link_libraries(system_test_import_tflite PRIVATE nnc_support mir_tflite_importer)
endif()

if(NNC_FRONTEND_CAFFE_ENABLED)
  add_executable(system_test_import_caffe caffe.cpp)
  target_link_libraries(system_test_import_caffe PRIVATE nnc_support mir_caffe_importer)
endif()
