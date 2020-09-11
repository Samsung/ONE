file(GLOB_RECURSE SOURCES "src/*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(onert_core SHARED ${SOURCES})
set_target_properties(onert_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(onert_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(onert_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(onert_core PUBLIC nnfw_lib_misc half)
target_link_libraries(onert_core PRIVATE nnfw_lib_cker)
target_link_libraries(onert_core PRIVATE nnfw_common)
target_link_libraries(onert_core PRIVATE nnfw_coverage)
target_link_libraries(onert_core PRIVATE dl ${LIB_PTHREAD})
target_link_libraries(onert_core PRIVATE jsoncpp)
target_link_libraries(onert_core PRIVATE ruy_instrumentation)
if (ANDROID)
  target_link_libraries(onert_core INTERFACE log)
endif (ANDROID)

if(ENVVAR_ONERT_CONFIG)
  target_compile_definitions(onert_core PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_ONERT_CONFIG)

install(TARGETS onert_core LIBRARY DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "include/onert"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.lst"
        )

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_BACKEND_CPU_COMMON test_onert_backend_cpu_common)

add_executable(${TEST_ONERT_BACKEND_CPU_COMMON} ${TESTS})

target_link_libraries(${TEST_ONERT_BACKEND_CPU_COMMON} onert_core)
target_link_libraries(${TEST_ONERT_BACKEND_CPU_COMMON} gtest gtest_main dl ${LIB_PTHREAD})

add_test(${TEST_ONERT_BACKEND_CPU_COMMON} ${TEST_ONERT_BACKEND_CPU_COMMON})
install(TARGETS ${TEST_ONERT_BACKEND_CPU_COMMON} DESTINATION unittest_standalone)
