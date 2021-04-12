if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS "src/*.test.cpp")

add_library(luci_testhelper STATIC ${TESTS})
target_include_directories(luci_testhelper PRIVATE src)
target_include_directories(luci_testhelper PUBLIC include)
target_link_libraries(luci_testhelper luci_lang)
