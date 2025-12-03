set(LIB_ONERT_BACKEND_TRIX onert_backend_trix)

nnfw_find_package(TRIXEngine QUIET)
if(NOT TRIXEngine_FOUND)
  message(STATUS "ONERT backend: Failed to find TRIXEngine")
  return()
endif(NOT TRIXEngine_FOUND)
message(STATUS "ONERT backend: Found TRIXEngine")

file(GLOB_RECURSE SOURCES "*.cc")

file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(${LIB_ONERT_BACKEND_TRIX} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE trix-engine)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE nnfw_coverage)
# Default TRIX_REQ_TIMEOUT_SEC is set as cmake option default setting
# If user want to change this value, please use cmake option -DTRIX_REQ_TIMEOUT_SEC=<value> when build. (ex: -DTRIX_REQ_TIMEOUT_SEC=100)
target_compile_definitions(${LIB_ONERT_BACKEND_TRIX} PRIVATE TIMEOUT_SEC=${TRIX_REQ_TIMEOUT_SEC})

set_target_properties(${LIB_ONERT_BACKEND_TRIX} PROPERTIES
  OUTPUT_NAME backend_trix
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

install(TARGETS ${LIB_ONERT_BACKEND_TRIX} DESTINATION ${ONERT_INSTALL_BACKENDDIR})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_TRIX_BACKEND test_onert_trix_backend)

add_executable(${TEST_ONERT_TRIX_BACKEND} ${TESTS})
target_link_libraries(${TEST_ONERT_TRIX_BACKEND} onert_core)
target_link_libraries(${TEST_ONERT_TRIX_BACKEND} trix-engine)
target_link_libraries(${TEST_ONERT_TRIX_BACKEND} ${LIB_ONERT_BACKEND_TRIX})
target_link_libraries(${TEST_ONERT_TRIX_BACKEND} nnfw_common)
target_link_libraries(${TEST_ONERT_TRIX_BACKEND} nnfw_coverage)
target_link_libraries(${TEST_ONERT_TRIX_BACKEND} gtest gtest_main Threads::Threads)
set_target_properties(${TEST_ONERT_TRIX_BACKEND} PROPERTIES
INSTALL_RPATH "$ORIGIN/../${ONERT_INSTALL_COREDIR}:$ORIGIN/../${ONERT_INSTALL_BACKENDDIR}")

add_test(${TEST_ONERT_TRIX_BACKEND} ${TEST_ONERT_TRIX_BACKEND})
install(TARGETS ${TEST_ONERT_TRIX_BACKEND} DESTINATION unittest)
