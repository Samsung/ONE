if(NOT INSTALL_TEST_SCRIPTS)
  return()
endif(NOT INSTALL_TEST_SCRIPTS)

# Install test driver
file(GLOB TEST_DRIVERS onert-test)
install(PROGRAMS ${TEST_DRIVERS} DESTINATION test)

# Install test scripts
install(DIRECTORY command DESTINATION test)
