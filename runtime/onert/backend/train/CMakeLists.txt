set(LIB_ONERT_BACKEND_TRAIN onert_backend_train)

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(${LIB_ONERT_BACKEND_TRAIN} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE ${LIB_ONERT_BACKEND_CPU})
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE nnfw_lib_cker nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_TRAIN} PROPERTIES
  OUTPUT_NAME backend_train
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../..")

install(TARGETS ${LIB_ONERT_BACKEND_TRAIN} DESTINATION lib/nnfw/backend)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_TRAIN_BACKEND test_onert_train_backend)

add_executable(${TEST_ONERT_TRAIN_BACKEND} ${TESTS})

target_link_libraries(${TEST_ONERT_TRAIN_BACKEND} ${LIB_ONERT_BACKEND_TRAIN})
# Requires linking nnfw_coverage: check header coverage
target_link_libraries(${TEST_ONERT_TRAIN_BACKEND} nnfw_coverage)
target_link_libraries(${TEST_ONERT_TRAIN_BACKEND} onert_core)
target_link_libraries(${TEST_ONERT_TRAIN_BACKEND} gtest gtest_main dl ${LIB_PTHREAD})

# Set install rpath to find onert_core, onert_backend_train, etc
set_target_properties(${TEST_ONERT_TRAIN_BACKEND} PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/nnfw/backend")

add_test(${TEST_ONERT_TRAIN_BACKEND} ${TEST_ONERT_TRAIN_BACKEND})
install(TARGETS ${TEST_ONERT_TRAIN_BACKEND} DESTINATION unittest)
