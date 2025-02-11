# from http://google.github.io/googletest/quickstart-cmake.html

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

function(GTest_AddTest TGT)
  add_executable(${TGT} ${ARGN})
  target_link_libraries(${TGT} GTest::gtest_main)
  if (ENABLE_COVERAGE)
    gtest_discover_tests(${TGT} XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/gtest_xml)
  else()
    gtest_discover_tests(${TGT})
  endif()
endfunction()

function(GTest_AddTest_Public TGT)
  add_executable(${TGT} ${ARGN})
  target_link_libraries(${TGT} PUBLIC GTest::gtest_main)
  if (ENABLE_COVERAGE)
    gtest_discover_tests(${TGT} XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/gtest_xml)
  else()
    gtest_discover_tests(${TGT})
  endif()
endfunction()
