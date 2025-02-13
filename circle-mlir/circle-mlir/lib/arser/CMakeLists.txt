add_library(arser INTERFACE)

# It specifies INTERFACE so that future targets linked with arser library will inherit its include directory.
# It means that a developer who want to link arser just need to add one line.
# target_link_library(another-users-target arser)
target_include_directories(arser INTERFACE include/)

# NOTE test for arser are removed.
#      instead, add arser_validate to validate header compilation.
add_library(arser_validate STATIC src/arser.cpp)
target_link_libraries(arser_validate PUBLIC arser)

if(NOT ENABLE_TEST)
  return()
endif()

GTest_AddTest(arser_test test/arser.test.cpp)
target_link_libraries(arser_test arser)
target_link_libraries(arser_test cirmlir_coverage)
