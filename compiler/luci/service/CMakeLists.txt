file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(luci_service ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_service PRIVATE src)
target_include_directories(luci_service PUBLIC include)
target_link_libraries(luci_service PUBLIC luci_lang)
target_link_libraries(luci_service PUBLIC mio_circle)
target_link_libraries(luci_service PUBLIC logo_core)
target_link_libraries(luci_service PRIVATE luci_log)
target_link_libraries(luci_service PRIVATE luci_logex)
target_link_libraries(luci_service PRIVATE nncc_common)
target_link_libraries(luci_service PRIVATE oops)
install(TARGETS luci_service DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_service_test ${TESTS})
target_include_directories(luci_service_test PRIVATE src)
target_link_libraries(luci_service_test luci_service)
target_link_libraries(luci_service_test luci_testhelper)
target_link_libraries(luci_service_test oops)
