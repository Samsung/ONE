if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(SRCS_TEST_DRIVER src/TestDriver.cpp)

# create driver
add_executable(test_driver ${SRCS_TEST_DRIVER})
target_link_libraries(test_driver PRIVATE luci_interpreter_import)
target_link_libraries(test_driver PRIVATE luci_interpreter)
target_link_libraries(test_driver PRIVATE safemain)

unset(EMBEDDED_IMPORT_VALUE_TESTS)

macro(addeval NAME)
  list(APPEND EMBEDDED_IMPORT_VALUE_TESTS ${NAME})
endmacro(addeval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

# Generate dependencies
add_custom_target(embedded_import_testfiles ALL DEPENDS ${TESTFILES})

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME embedded_import_value_test
         COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify.sh" "${CMAKE_CURRENT_BINARY_DIR}" "${ARTIFACTS_BIN_PATH}"
                 ${EMBEDDED_IMPORT_VALUE_TESTS})
