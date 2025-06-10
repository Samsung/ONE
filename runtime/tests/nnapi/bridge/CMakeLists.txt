file(GLOB_RECURSE SOURCES_FRONTEND "*.cc")
file(GLOB_RECURSE TESTS_FRONTEND "*.test.cc")
list(REMOVE_ITEM SOURCES_FRONTEND ${TESTS_FRONTEND})

set(LIB_NNAPI onert_nnapi)

add_library(${LIB_NNAPI} STATIC ${SOURCES_FRONTEND})
target_include_directories(${LIB_NNAPI} PUBLIC include)
target_link_libraries(${LIB_NNAPI} PUBLIC onert_core) # TODO Link PRIVATE onert_core
target_link_libraries(${LIB_NNAPI} PRIVATE nnfw_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(test_onert_frontend_nnapi ${TESTS_FRONTEND})

target_link_libraries(test_onert_frontend_nnapi ${LIB_NNAPI} dl)
target_link_libraries(test_onert_frontend_nnapi gtest)
target_link_libraries(test_onert_frontend_nnapi gtest_main)
# Set INSTALL_RPATH to find onert_core
set_target_properties(test_onert_frontend_nnapi PROPERTIES INSTALL_RPATH "$ORIGIN/../lib/nnfw")

install(TARGETS test_onert_frontend_nnapi DESTINATION unittest)
