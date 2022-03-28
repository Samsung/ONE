nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build dio_hdf5: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(dio_hdf5 SHARED ${SOURCES})
target_include_directories(dio_hdf5 PUBLIC include)
target_include_directories(dio_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(dio_hdf5 ${HDF5_CXX_LIBRARIES})
target_link_libraries(dio_hdf5 loco)

install(TARGETS dio_hdf5 DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(dio_hdf5_test ${TESTS} "src/HDF5Importer.cpp")
target_link_libraries(dio_hdf5_test dio_hdf5)
