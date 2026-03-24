set(interp_src InterpreterBackend.cpp)
nnc_add_library(nnc_interpreter SHARED ${interp_src})
target_link_libraries(nnc_interpreter PRIVATE mir_interpreter)

if(NNC_HDF5_SUPPORTED)
  target_include_directories(nnc_interpreter PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(nnc_interpreter PRIVATE ${HDF5_CXX_LIBRARIES})
endif(NNC_HDF5_SUPPORTED)

# install interpreter library
nnc_install_library(nnc_interpreter)
