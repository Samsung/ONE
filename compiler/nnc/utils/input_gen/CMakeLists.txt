find_package(
  HDF5
  COMPONENTS CXX
  QUIET)

if(NOT HDF5_FOUND)
  return()
endif(NOT HDF5_FOUND)

add_executable(tensor_gen tensor_gen.cpp)
target_include_directories(tensor_gen PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(tensor_gen ${HDF5_CXX_LIBRARIES})
