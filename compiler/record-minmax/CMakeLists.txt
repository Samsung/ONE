set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(record-minmax ${DRIVER} ${SOURCES})
target_include_directories(record-minmax PRIVATE include)

target_link_libraries(record-minmax arser)
target_link_libraries(record-minmax safemain)
target_link_libraries(record-minmax luci_import)
target_link_libraries(record-minmax luci_env)
target_link_libraries(record-minmax luci_export)
target_link_libraries(record-minmax luci_interpreter)
target_link_libraries(record-minmax dio_hdf5)
target_link_libraries(record-minmax vconone)
target_link_libraries(record-minmax nncc_coverage)

find_package(OpenMP REQUIRED)

target_link_libraries(record-minmax OpenMP::OpenMP_CXX)

install(TARGETS record-minmax DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

file(GLOB_RECURSE TESTS "tests/*.test.cpp")

nnas_find_package(GTest REQUIRED)
GTest_AddTest(record_minmax_function_test "${TESTS}")
target_include_directories(record_minmax_function_test PRIVATE include)
target_link_libraries(record_minmax_function_test nncc_coverage)
