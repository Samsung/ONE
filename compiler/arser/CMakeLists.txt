add_library(arser INTERFACE)

# It specifies INTERFACE so that future targets linked with arser library will inherit its include directory.
# It means that a developer who want to link arser just need to add one line.
# target_link_library(another-users-target arser)
target_include_directories(arser INTERFACE include/)
target_link_libraries(arser INTERFACE nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)
set(TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/arser.test.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/HelpMessage.test.cpp")
GTest_AddTest(arser_test ${TESTS})
target_link_libraries(arser_test arser)
