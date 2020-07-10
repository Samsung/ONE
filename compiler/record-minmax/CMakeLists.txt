find_package(HDF5 COMPONENTS CXX QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build record-minmax: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(record-minmax ${DRIVER} ${SOURCES})
target_include_directories(record-minmax PRIVATE include)
target_include_directories(record-minmax PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(record-minmax ${HDF5_CXX_LIBRARIES})
target_link_libraries(record-minmax arser)
target_link_libraries(record-minmax safemain)
target_link_libraries(record-minmax luci_import)
target_link_libraries(record-minmax luci_export)
target_link_libraries(record-minmax luci_interpreter)

install(TARGETS record-minmax DESTINATION bin)

nnas_find_package(GTest REQUIRED)
GTest_AddTest(record_minmax_function_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/RecordFunction.test.cpp")
target_include_directories(record_minmax_function_test PRIVATE include)
