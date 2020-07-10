if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)

nnas_find_package(HDF5 QUIET)

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
