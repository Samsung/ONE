if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)

# TODO This will be nnas_find_package
find_package(HDF5 COMPONENTS CXX QUIET CONFIG)

if(NOT HDF5_FOUND)
  find_package(HDF5 COMPONENTS CXX QUIET MODULE)
endif(NOT HDF5_FOUND)

if(NOT HDF5_FOUND)
  return()
endif(NOT HDF5_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-tensordump ${DRIVER} ${SOURCES})
target_include_directories(circle-tensordump PRIVATE src)
target_include_directories(circle-tensordump PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(circle-tensordump PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(circle-tensordump PRIVATE arser)
target_link_libraries(circle-tensordump PRIVATE foder)
target_link_libraries(circle-tensordump PRIVATE mio_circle)
target_link_libraries(circle-tensordump PRIVATE safemain)
target_link_libraries(circle-tensordump PRIVATE stdex)
