file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(record-hessian STATIC ${SOURCES})

target_include_directories(record-hessian PUBLIC include)
target_include_directories(record-hessian PRIVATE src)

target_link_libraries(record-hessian luci_import)
target_link_libraries(record-hessian luci_env)
target_link_libraries(record-hessian luci_export)
target_link_libraries(record-hessian luci_interpreter)
target_link_libraries(record-hessian luci_log)
target_link_libraries(record-hessian dio_hdf5)
target_link_libraries(record-hessian nncc_common)

install(TARGETS record-hessian DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(record_hessian_tests ${TESTS})
target_include_directories(record_hessian_tests PRIVATE include)
target_include_directories(record_hessian_tests PRIVATE src)
target_link_libraries(record_hessian_tests luci_lang)
target_link_libraries(record_hessian_tests luci_pass)
target_link_libraries(record_hessian_tests loco)
target_link_libraries(record_hessian_tests dio_hdf5)
target_link_libraries(record_hessian_tests nncc_coverage)
target_link_libraries(record_hessian_tests luci_interpreter)
target_link_libraries(record_hessian_tests record-hessian)
