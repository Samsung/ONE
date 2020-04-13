file(GLOB_RECURSE SOURCES_FRONTEND "*.cc")
file(GLOB_RECURSE TESTS_FRONTEND "*.test.cc")
list(REMOVE_ITEM SOURCES_FRONTEND ${TESTS_FRONTEND})

set(LIB_NEURUN neurun)

add_library(${LIB_NEURUN} SHARED ${SOURCES_FRONTEND})
target_link_libraries(${LIB_NEURUN} PUBLIC nnfw-nnapi-header)
target_link_libraries(${LIB_NEURUN} PUBLIC neurun_core) # TODO Link PRIVATE neurun_core
target_link_libraries(${LIB_NEURUN} PRIVATE nnfw_common)
target_link_libraries(${LIB_NEURUN} PRIVATE nnfw_coverage)

set_target_properties(${LIB_NEURUN} PROPERTIES OUTPUT_NAME neuralnetworks)

install(TARGETS ${LIB_NEURUN} DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(test_neurun_frontend_nnapi ${TESTS_FRONTEND})

target_link_libraries(test_neurun_frontend_nnapi PRIVATE ${LIB_NEURUN} dl)
target_link_libraries(test_neurun_frontend_nnapi PRIVATE gtest)
target_link_libraries(test_neurun_frontend_nnapi PRIVATE gtest_main)

install(TARGETS test_neurun_frontend_nnapi DESTINATION unittest)
