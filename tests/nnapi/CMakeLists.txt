if (NOT BUILD_RUNTIME_NNAPI_TEST)
  return()
endif(NOT BUILD_RUNTIME_NNAPI_TEST)

if (NOT BUILD_ONERT)
  message(STATUS "Skip build NNAPI test: no runtime build")
  return()
endif(NOT BUILD_ONERT)

nnfw_find_package(GTest)

set(GENERATED_CPPS "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/all_generated_V1_2_cts_tests.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/all_generated_V1_1_cts_tests.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/all_generated_V1_0_cts_tests.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/all_generated_Ex_cts_tests.cpp")

# Skip script for generate test and use pre-generated test
if (GENERATE_RUNTIME_NNAPI_TESTS)
  set(GENERATOR_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/specs/generate_test.sh")
  file(GLOB_RECURSE TEST_SPECS "${CMAKE_CURRENT_SOURCE_DIR}/specs/*.mod.py")

  add_custom_command(OUTPUT ${GENERATED_CPPS}
                     COMMAND ${GENERATOR_SCRIPT}
                     DEPENDS ${GENERATOR_SCRIPT} ${TEST_SPECS}
                     COMMENT "Generating runtime nnapi tests")
  add_custom_target(generate-runtime-nnapi-tests
                    DEPENDS ${GENERATED_CPPS})
endif(GENERATE_RUNTIME_NNAPI_TESTS)

set(RUNTIME_NNAPI_TEST nnapi_gtest)
set(RUNTIME_NNAPI_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/TestMain.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/src/TestNeuralNetworksWrapper.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/src/TestTrivialModel.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/src/TestValidation.cpp)
set(GENERATED_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/TestGenerated.cpp
                       ${GENERATED_CPPS})
set(RUNTIME_NNAPI_TEST_SRC ${RUNTIME_NNAPI_TEST_SRC} ${GENERATED_TEST_SRC})

add_executable(${RUNTIME_NNAPI_TEST} ${RUNTIME_NNAPI_TEST_SRC})

if (GENERATE_RUNTIME_NNAPI_TESTS)
  add_dependencies(${RUNTIME_NNAPI_TEST} generate-runtime-nnapi-tests)
endif(GENERATE_RUNTIME_NNAPI_TESTS)

set(RUNTIME_NNAPI_TEST_SRC_INC ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${RUNTIME_NNAPI_TEST} PRIVATE ${RUNTIME_NNAPI_TEST_SRC_INC})

# Define NNTEST_ONLY_PUBLIC_API to avoid android dependency
target_compile_definitions(${RUNTIME_NNAPI_TEST} PRIVATE NNTEST_ONLY_PUBLIC_API)

target_link_libraries(${RUNTIME_NNAPI_TEST} nnfw_lib_nnapi)
target_link_libraries(${RUNTIME_NNAPI_TEST} gtest gmock)
target_link_libraries(${RUNTIME_NNAPI_TEST} ${LIB_PTHREAD} dl)

install(TARGETS ${RUNTIME_NNAPI_TEST} DESTINATION nnapi-gtest)

# Default test backend: cpu
set(SKIPLIST_FILE_NAME ${RUNTIME_NNAPI_TEST}.skip.${TARGET_PLATFORM}.cpu)
install(FILES ${SKIPLIST_FILE_NAME}
        DESTINATION nnapi-gtest
        RENAME ${RUNTIME_NNAPI_TEST}.skip
        OPTIONAL)

# Install skiplist file for target as backup
FILE(GLOB SKIPLIST_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/${RUNTIME_NNAPI_TEST}.skip.${TARGET_PLATFORM}*)
install(FILES ${SKIPLIST_TARGET} DESTINATION nnapi-gtest OPTIONAL)
