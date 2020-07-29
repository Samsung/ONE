if(NOT INSTALL_TEST_SCRIPTS)
  return()
endif(NOT INSTALL_TEST_SCRIPTS)

# Install test driver
file(GLOB TEST_DRIVERS onert-test)
install(PROGRAMS ${TEST_DRIVERS} DESTINATION bin)

# Install test scripts
file(GLOB TEST_SCRIPTS command/unittest)
install(FILES ${TEST_SCRIPTS} DESTINATION tests)
