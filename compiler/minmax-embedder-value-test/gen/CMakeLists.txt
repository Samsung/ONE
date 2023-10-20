# Build minmax-data-gen
nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build minmax-datagen: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(minmax_data_gen "${SOURCES}")
target_include_directories(minmax_data_gen PUBLIC ${HDF5_INCLUDE_DIRS})

target_link_libraries(minmax_data_gen ${HDF5_CXX_LIBRARIES})
target_link_libraries(minmax_data_gen safemain)
target_link_libraries(minmax_data_gen arser)
