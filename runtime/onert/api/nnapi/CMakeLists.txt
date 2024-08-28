# NNAPI Frontend is only for test purpose.
# So, it's not built by default.
if(NOT ENABLE_TEST OR NOT BUILD_RUNTIME_NNAPI_TEST)
  return()
endif(NOT ENABLE_TEST OR NOT BUILD_RUNTIME_NNAPI_TEST)

file(GLOB_RECURSE SOURCES_FRONTEND "*.cc")
file(GLOB_RECURSE TESTS_FRONTEND "*.test.cc")
list(REMOVE_ITEM SOURCES_FRONTEND ${TESTS_FRONTEND})

set(LIB_ONERT_NNAPI onert-nnapi)

add_library(${LIB_ONERT_NNAPI} SHARED ${SOURCES_FRONTEND})
target_include_directories(${LIB_ONERT_NNAPI} PUBLIC include)
target_link_libraries(${LIB_ONERT_NNAPI} PUBLIC onert_core) # TODO Link PRIVATE onert_core
target_link_libraries(${LIB_ONERT_NNAPI} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_NNAPI} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_NNAPI} PROPERTIES OUTPUT_NAME neuralnetworks)

install(TARGETS ${LIB_ONERT_NNAPI} DESTINATION lib)

add_executable(test_onert_frontend_nnapi ${TESTS_FRONTEND})

target_link_libraries(test_onert_frontend_nnapi PRIVATE ${LIB_ONERT_NNAPI} dl)
target_link_libraries(test_onert_frontend_nnapi PRIVATE gtest)
target_link_libraries(test_onert_frontend_nnapi PRIVATE gtest_main)

install(TARGETS test_onert_frontend_nnapi DESTINATION unittest)
