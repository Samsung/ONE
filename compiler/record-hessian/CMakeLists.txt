file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(record-hessian STATIC ${SOURCES})

target_include_directories(record-hessian PUBLIC include)
target_include_directories(record-hessian PRIVATE src)
target_link_libraries(record-hessian luci_import)
target_link_libraries(record-hessian luci_env)
target_link_libraries(record-hessian luci_export)
target_link_libraries(record-hessian luci_interpreter)
target_link_libraries(record-hessian luci_log)
target_link_libraries(record-hessian dio_hdf5)
install(TARGETS record-hessian DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)
set(TEST_SOURCES, "src/MinMaxComputer.cpp")

file(GLOB_RECURSE TESTS "tests/*.test.cpp")

GTest_AddTest(record_hessian_tests ${TESTS} ${TEST_SOURCES})
# target_include_directories(record_hessian_tests PRIVATE include)
target_include_directories(record_hessian_tests PUBLIC include)
target_include_directories(record_hessian_tests PRIVATE src)
target_link_libraries(record_hessian_tests luci_lang)
target_link_libraries(record_hessian_tests nncc_coverage)
target_link_libraries(record_hessian_tests luci_interpreter)