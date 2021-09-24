file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(luci_lang ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_lang PRIVATE src)
target_include_directories(luci_lang PUBLIC include)
target_link_libraries(luci_lang PUBLIC loco)
target_link_libraries(luci_lang PUBLIC oops)
target_link_libraries(luci_lang PUBLIC nncc_coverage)
target_link_libraries(luci_lang PRIVATE logo)
target_link_libraries(luci_lang PRIVATE nncc_common)

install(TARGETS luci_lang DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_lang_test ${TESTS})
target_include_directories(luci_lang_test PRIVATE src)
target_link_libraries(luci_lang_test luci_lang)
target_link_libraries(luci_lang_test logo)
