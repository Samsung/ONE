if(NOT TARGET mio_circle09)
  return()
endif(NOT TARGET mio_circle09)

nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build circle-tensordump: FAILED (missing HDF5)")
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
target_link_libraries(circle-tensordump PRIVATE mio_circle09)
target_link_libraries(circle-tensordump PRIVATE mio_circle09_helper)
target_link_libraries(circle-tensordump PRIVATE safemain)

install(TARGETS circle-tensordump DESTINATION bin)
