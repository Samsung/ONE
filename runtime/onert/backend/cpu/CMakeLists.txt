nnfw_find_package(Ruy REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(${LIB_ONERT_BACKEND_CPU} SHARED ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_CPU} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_lib_cker nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ruy)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} INTERFACE ruy_instrumentation)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ndarray)
# Set public: ExternalContext is used in train backend
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PUBLIC ggml)

set_target_properties(${LIB_ONERT_BACKEND_CPU} PROPERTIES
  OUTPUT_NAME backend_cpu
  POSITION_INDEPENDENT_CODE ON
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

install(TARGETS ${LIB_ONERT_BACKEND_CPU} DESTINATION ${ONERT_INSTALL_BACKENDDIR})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_CPU_BACKEND test_onert_cpu_backend)

add_executable(${TEST_ONERT_CPU_BACKEND} ${TESTS})

target_link_libraries(${TEST_ONERT_CPU_BACKEND} ${LIB_ONERT_BACKEND_CPU})
# Requires linking nnfw_coverage: check header coverage
target_link_libraries(${TEST_ONERT_CPU_BACKEND} nnfw_coverage)
target_link_libraries(${TEST_ONERT_CPU_BACKEND} onert_core)
target_link_libraries(${TEST_ONERT_CPU_BACKEND} gtest gtest_main dl ${LIB_PTHREAD})

# Set install rpath to find onert_core, onert_backend_cpu, etc
set_target_properties(${TEST_ONERT_CPU_BACKEND} PROPERTIES
  INSTALL_RPATH "$ORIGIN/../${ONERT_INSTALL_COREDIR}:$ORIGIN/../${ONERT_INSTALL_BACKENDDIR}")

add_test(${TEST_ONERT_CPU_BACKEND} ${TEST_ONERT_CPU_BACKEND})
install(TARGETS ${TEST_ONERT_CPU_BACKEND} DESTINATION unittest)
