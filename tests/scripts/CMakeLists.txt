if(NOT INSTALL_TEST_SCRIPTS)
  return()
endif(NOT INSTALL_TEST_SCRIPTS)

# Install test driver
file(GLOB TEST_DRIVER_SCRIPT onert-test)
install(PROGRAMS ${TEST_DRIVER_SCRIPT} DESTINATION test)

# Commands don't have execute permission itself
install(DIRECTORY command DESTINATION test)

# Install models test script
file(GLOB MODEL_TEST_SCRIPT "models/run_test.sh")
install(PROGRAMS ${MODEL_TEST_SCRIPT} DESTINATION test/models)

# Install models test list file
file(GLOB MODEL_TEST_DIR models/config)
install(DIRECTORY ${MODEL_TEST_DIR} DESTINATION test/models)

# Install nnpackage test config
file(GLOB MODEL_TEST_DIR LIST_DIRECTORIES true nnfw_api_gtest/models/*)
install(DIRECTORY ${MODEL_TEST_DIR} DESTINATION test/models/nnpackage)

# Install test list
file(GLOB TEST_LIST_DIR list)
install(DIRECTORY ${TEST_LIST_DIR} DESTINATION test)
