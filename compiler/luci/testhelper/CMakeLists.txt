if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

# NOTE we are using "*.test.cpp" NOT to be included in static analyzer tools

# testhelper library itself
set(HELPER_SOURCE
      src/TestShape.test.cpp
   )

add_library(luci_testhelper STATIC ${HELPER_SOURCE})
target_include_directories(luci_testhelper PRIVATE src)
target_include_directories(luci_testhelper PUBLIC include)
target_link_libraries(luci_testhelper luci_lang)

# test for testhelper library
set(TESTER_SOURCE
      src/TestIOGraph.test.cpp
   )

GTest_AddTest(luci_testhelper_test ${TESTER_SOURCE})
target_link_libraries(luci_testhelper_test luci_testhelper)
