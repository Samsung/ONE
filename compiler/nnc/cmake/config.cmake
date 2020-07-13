#
# definition of directories of all nnc component
#
set(NNC_DRIVER_DIR ${NNC_ROOT_SRC_DIR}/driver)
set(NNC_SOFT_BACKEND_DIR ${NNC_ROOT_SRC_DIR}/backends/soft_backend)
set(NNC_ACL_BACKEND_DIR ${NNC_ROOT_SRC_DIR}/backends/acl_soft_backend)
set(NNC_INTERPRETER_DIR ${NNC_ROOT_SRC_DIR}/backends/interpreter)
set(NNC_SUPPORT_DIR ${NNC_ROOT_SRC_DIR}/support)
set(NNC_PASS_DIR ${NNC_ROOT_SRC_DIR}/pass)

#
# Other additional useful cmake variables
#
set(NNC_ENABLE_UNITTEST ${ENABLE_TEST})
set(NNC_TARGET_EXECUTABLE nnc) # nnc main target

set(NNC_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}) # root path of installation directory
set(NNC_INSTALL_BIN_PATH ${NNC_INSTALL_PATH}/bin)
set(NNC_INSTALL_LIB_PATH ${NNC_INSTALL_PATH}/lib) # directory that contains other directories with shared library

#
# find necessary packages
#
# TODO This will be nnas_find_package
find_package(HDF5 COMPONENTS CXX QUIET CONFIG)

if(NOT HDF5_FOUND)
  find_package(HDF5 COMPONENTS CXX QUIET MODULE)
endif(NOT HDF5_FOUND)

# defines if hdf5 package was found
if(HDF5_FOUND)
  set(NNC_HDF5_SUPPORTED ON)
else()
  message(WARNING "HDF5 not found, functionality of some nnc components will be disabled")
  set(NNC_HDF5_SUPPORTED OFF)
endif()

if(TARGET mir_caffe2_importer)
  set(NNC_FRONTEND_CAFFE2_ENABLED ON)
else()
  set(NNC_FRONTEND_CAFFE2_ENABLED OFF)
endif()

if(TARGET mir_caffe_importer)
  set(NNC_FRONTEND_CAFFE_ENABLED ON)
else()
  set(NNC_FRONTEND_CAFFE_ENABLED OFF)
endif()

if(TARGET mir_onnx_importer)
  set(NNC_FRONTEND_ONNX_ENABLED ON)
else()
  set(NNC_FRONTEND_ONNX_ENABLED OFF)
endif()

if(TARGET mir_tflite_importer)
  set(NNC_FRONTEND_TFLITE_ENABLED ON)
else()
  set(NNC_FRONTEND_TFLITE_ENABLED OFF)
endif()
