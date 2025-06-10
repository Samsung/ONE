if(NOT ENABLE_TEST)
  return()
endif()

set(SRC_TEST
  CheckGTest.test.cpp
)

GTest_AddTest(check_gtest ${SRC_TEST})
