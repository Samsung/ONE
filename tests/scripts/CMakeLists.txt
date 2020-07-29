if(NOT INSTALL_TEST_SCRIPTS)
  return()
endif(NOT INSTALL_TEST_SCRIPTS)

# Install test driver
file(GLOB TEST_DRIVERS onert-test)
install(PROGRAMS ${TEST_DRIVERS} DESTINATION bin)

# Install test scripts
file(GLOB TEST_SCRIPTS "*.sh")
install(PROGRAMS ${TEST_SCRIPTS} DESTINATION tests/scripts)

# Install test list
file(GLOB TEST_LISTS "list/*.txt")
install(FILES ${TEST_LISTS} DESTINATION tests/scripts/list)

# Install framework test script
file(GLOB FRAMEWORKTEST_SCRIPT "framework/run_test.sh")
install(PROGRAMS ${FRAMEWORKTEST_SCRIPT} DESTINATION tests/scripts/framework)

# Install framework test list file
file(GLOB FRAMEWORKTEST_DIR framework/tests)
install(DIRECTORY ${FRAMEWORKTEST_DIR} DESTINATION tests/scripts/framework)
