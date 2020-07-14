# TODO This will be nnas_find_package
find_package(HDF5 COMPONENTS CXX QUIET CONFIG)

if(NOT HDF5_FOUND)
  find_package(HDF5 COMPONENTS CXX QUIET MODULE)
endif(NOT HDF5_FOUND)

if(NOT HDF5_FOUND)
  return()
endif(NOT HDF5_FOUND)

nnas_find_package(TensorFlow QUIET)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(gen_core STATIC ${SOURCES})
set_target_properties(gen_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gen_core PUBLIC include)
target_include_directories(gen_core PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gen_core ${HDF5_CXX_LIBRARIES})
target_link_libraries(gen_core tfinfo_v2)
target_link_libraries(gen_core angkor)
