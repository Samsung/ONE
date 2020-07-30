if(NOT INSTALL_TEST_SCRIPTS)
  return()
endif(NOT INSTALL_TEST_SCRIPTS)

# Install test driver
file(GLOB TEST_DRIVER_SCRIPT onert-test)
install(PROGRAMS ${TEST_DRIVER_SCRIPT} DESTINATION test)

# TODO Below install to tests/ will be deprecated

# Install test scripts
file(GLOB TEST_SCRIPTS "*.sh")
install(PROGRAMS ${TEST_SCRIPTS} DESTINATION tests/scripts)

# Install test list
file(GLOB TEST_LISTS "list/*.txt")
install(FILES ${TEST_LISTS} DESTINATION tests/scripts/list)

# Install models test script
file(GLOB MODEL_TEST_SCRIPT "models/run_test.sh")
install(PROGRAMS ${MODEL_TEST_SCRIPT} DESTINATION tests/scripts/models)

# Install models test list file
file(GLOB MODEL_TEST_DIR models/config)
install(DIRECTORY ${MODEL_TEST_DIR} DESTINATION tests/scripts/models)
