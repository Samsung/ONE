if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(LUCI_VALUE_TESTS)
unset(LUCI_VALUE_TESTS_TOL)

macro(addeval NAME)
  list(APPEND LUCI_VALUE_TESTS ${NAME})
endmacro(addeval)

macro(addevaltol NAME RTOL ATOL)
  list(APPEND LUCI_VALUE_TESTS_TOL ${NAME} ${RTOL} ${ATOL})
endmacro(addevaltol)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

# Generate dependencies
add_custom_target(luci_eval_testfiles ALL DEPENDS ${TESTFILES})

if(NOT CMAKE_CROSSCOMPILING)

  get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

  add_test(NAME luci_value_test
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify.sh"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${ARTIFACTS_BIN_PATH}"
            "${NNCC_OVERLAY_DIR}/venv_2_12_1"
            "$<TARGET_FILE:luci_eval_driver>"
            ${LUCI_VALUE_TESTS}
  )

  if(DEFINED LUCI_VALUE_TESTS_TOL)
    add_test(NAME luci_value_tol_test
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverifytol.sh"
              "${CMAKE_CURRENT_BINARY_DIR}"
              "${ARTIFACTS_BIN_PATH}"
              "${NNCC_OVERLAY_DIR}/venv_2_12_1"
              "$<TARGET_FILE:luci_eval_driver>"
              ${LUCI_VALUE_TESTS_TOL}
    )
  endif()

else(NOT CMAKE_CROSSCOMPILING)
  # NOTE target test is carried out using reference input/output data from host
  #      test results. this is because it would be difficult to prepare
  #      TensorFlow lite for target device.
  #      thus, one must run the host test and then run the test in target device
  #      with the test result files from the host test.

  if(NOT DEFINED ENV{BUILD_HOST_EXEC})
    message(STATUS "BUILD_HOST_EXEC not set: Skip luci-value-test")
    return()
  endif(NOT DEFINED ENV{BUILD_HOST_EXEC})

  set(ARTIFACTS_BIN_PATH $ENV{BUILD_HOST_EXEC}/compiler/common-artifacts)

  add_test(NAME luci_value_cross_test
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify_ref.sh"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${ARTIFACTS_BIN_PATH}"
            "$<TARGET_FILE:luci_eval_driver>"
            ${LUCI_VALUE_TESTS}
  )

  if(DEFINED LUCI_VALUE_TESTS_TOL)
    add_test(NAME luci_value_cross_tol_test
             COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverifytol_ref.sh"
                     "${CMAKE_CURRENT_BINARY_DIR}"
                     "${ARTIFACTS_BIN_PATH}"
                     "$<TARGET_FILE:luci_eval_driver>"
                     ${LUCI_VALUE_TESTS_TOL}
    )
  endif()

endif(NOT CMAKE_CROSSCOMPILING)
