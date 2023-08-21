if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

# Disable the test if record-minmax-for-thread-test does not exist
if (NOT TARGET record-minmax-for-thread-test)
    message(STATUS "record-minmax-thread-safety-test is disabled as record-minmax-for-thread-test was not built.")
    return()
endif(NOT TARGET record-minmax-for-thread-test)

# Build record-minmax-for-thread-test if target arch is 64bit
# Thread sanitizer is only available on 64bit machine
# (https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual#supported-platforms)
if(NOT "${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    return()
endif(NOT "${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")

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
        "${NNCC_OVERLAY_DIR}/venv_2_12_1"
        ${RECORD_MINMAX_THREAD_SAFETY_TEST}
)

if(ONE_UBUNTU_CODENAME_JAMMY)
    add_test(
            NAME record_minmax_thread_safety_210_test
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh"
            "${TEST_CONFIG}"
            "${ARTIFACTS_BIN_PATH}"
            "${NNCC_OVERLAY_DIR}/venv_2_10_1"
            ${RECORD_MINMAX_THREAD_SAFETY_TEST}
    )
endif(ONE_UBUNTU_CODENAME_JAMMY)
