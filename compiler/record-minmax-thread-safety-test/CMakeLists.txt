if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

unset(RECORD_MINMAX_THREAD_SAFETY_TEST)

macro(addTest NAME)
    list(APPEND RECORD_MINMAX_THREAD_SAFETY_TEST ${NAME})
endmacro(addTest)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
        OUTPUT "${MICRO_ARM_BUILD_DEPENDENCY}"
        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/standalone" ${CMAKE_ARM_OPTIONS}
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/standalone/CMakeLists.txt"
        VERBATIM
)
set(RECORD_MINMAX_PATH "$<TARGET_FILE:record-minmax-for-thread-test>")

add_custom_command(
        OUTPUT ${TEST_CONFIG}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
        COMMAND ${CMAKE_COMMAND} -E echo 'RECORD_MINMAX_PATH=\"$<TARGET_FILE:record-minmax-for-thread-test>\"' >> ${TEST_CONFIG}
        DEPENDS record-minmax-for-thread-test
        COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(record_minmax_thread_safety_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
        NAME record_minmax_thread_safety_test
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh"
        "${TEST_CONFIG}"
        "${ARTIFACTS_BIN_PATH}"
        "${NNCC_OVERLAY_DIR}/venv_2_8_0"
        ${RECORD_MINMAX_THREAD_SAFETY_TEST}
)
