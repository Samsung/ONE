file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_executable(circle-mpqsolver "${SOURCES}")
target_link_libraries(circle-mpqsolver arser)
target_link_libraries(circle-mpqsolver vconone)
target_link_libraries(circle-mpqsolver safemain)
target_link_libraries(circle-mpqsolver luci_lang)
target_link_libraries(circle-mpqsolver luci_service)
target_link_libraries(circle-mpqsolver luci_pass)

install(TARGETS circle-mpqsolver DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# circle-mpqsolver is executable, so we do not link it to the test.
# Instead, we use TEST_SOURCES to specify sources uesd for tests.
set(TEST_SOURCES
    "src/bisection/DepthParameterizer.cpp"
    "src/bisection/Quantizer.cpp")

nnas_find_package(GTest REQUIRED)
GTest_AddTest(circle_mpqsolver_test ${TESTS} ${TEST_SOURCES})
target_link_libraries(circle_mpqsolver_test luci_lang)
target_link_libraries(circle_mpqsolver_test luci_service)
target_link_libraries(circle_mpqsolver_test luci_pass)

