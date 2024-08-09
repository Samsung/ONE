file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

if (NOT LUCI_LIBRARY_TYPE)
  set(LUCI_LIBRARY_TYPE "SHARED")
endif(NOT LUCI_LIBRARY_TYPE)

add_library(luci_partition ${LUCI_LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_partition PRIVATE src)
target_include_directories(luci_partition PUBLIC include)
target_link_libraries(luci_partition PUBLIC luci_lang)
target_link_libraries(luci_partition PRIVATE luci_service)
target_link_libraries(luci_partition PRIVATE luci_log)
target_link_libraries(luci_partition PRIVATE luci_logex)
target_link_libraries(luci_partition PRIVATE mio_circle09)
target_link_libraries(luci_partition PRIVATE nncc_common)
target_link_libraries(luci_partition PRIVATE pepper_csv2vec)
target_link_libraries(luci_partition PRIVATE oops)

install(TARGETS luci_partition DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_partition_test ${TESTS})
target_include_directories(luci_partition_test PRIVATE src)
target_link_libraries(luci_partition_test luci_lang)
target_link_libraries(luci_partition_test luci_partition)
target_link_libraries(luci_partition_test luci_testhelper)
target_link_libraries(luci_partition_test luci_service)
