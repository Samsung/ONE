nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build h5_importer: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(h5_importer SHARED ${SOURCES})
target_include_directories(h5_importer PUBLIC include)
target_include_directories(h5_importer PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(h5_importer ${HDF5_CXX_LIBRARIES})
target_link_libraries(h5_importer loco)

install(TARGETS h5_importer DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
