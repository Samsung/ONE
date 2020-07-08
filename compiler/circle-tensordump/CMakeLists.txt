if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)
set(HDF5_USE_STATIC_LIBRARIES OFF)
unset(HDF5_CXX_LIBRARY_hdf5 CACHE)
unset(HDF5_CXX_LIBRARY_hdf5_cpp CACHE)
find_package(HDF5 COMPONENTS CXX QUIET)

if(NOT HDF5_FOUND)
  return()
endif(NOT HDF5_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-tensordump ${DRIVER} ${SOURCES})
target_include_directories(circle-tensordump PRIVATE src)
target_include_directories(circle-tensordump PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(circle-tensordump PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(circle-tensordump PRIVATE foder)
target_link_libraries(circle-tensordump PRIVATE mio_circle)
target_link_libraries(circle-tensordump PRIVATE safemain)
target_link_libraries(circle-tensordump PRIVATE stdex)
