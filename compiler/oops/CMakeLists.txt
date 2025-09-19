add_library(oops INTERFACE)
target_include_directories(oops INTERFACE include)
target_link_libraries(oops INTERFACE pepper_str)
target_link_libraries(oops INTERFACE nncc_coverage)

# oops is used by many other libraries' public header files,
# so its header should be installed to ensure their completeness
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(oops_test src/oops.test.cpp)
target_link_libraries(oops_test oops)
