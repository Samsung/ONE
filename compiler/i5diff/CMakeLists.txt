set(HDF5_USE_STATIC_LIBRARIES OFF)
unset(HDF5_CXX_LIBRARY_hdf5 CACHE)
unset(HDF5_CXX_LIBRARY_hdf5_cpp CACHE)
find_package(HDF5 COMPONENTS CXX QUIET)

if(NOT HDF5_FOUND)
  return()
endif(NOT HDF5_FOUND)

message(STATUS "Enable i5diff: TRUE")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(i5diff ${SOURCES})
target_include_directories(i5diff PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(i5diff PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(i5diff PRIVATE angkor)
target_link_libraries(i5diff PRIVATE safemain)
