file(GLOB_RECURSE SOURCES_FRONTEND "*.cc")
file(GLOB_RECURSE TESTS_FRONTEND "*.test.cc")
list(REMOVE_ITEM SOURCES_FRONTEND ${TESTS_FRONTEND})

set(LIB_ONERT onert)

add_library(${LIB_ONERT} SHARED ${SOURCES_FRONTEND})
target_link_libraries(${LIB_ONERT} PUBLIC nnfw-nnapi-header)
target_link_libraries(${LIB_ONERT} PUBLIC onert_core) # TODO Link PRIVATE onert_core
target_link_libraries(${LIB_ONERT} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT} PROPERTIES OUTPUT_NAME neuralnetworks)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT}>)
endif()

install(TARGETS ${LIB_ONERT} DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(test_onert_frontend_nnapi ${TESTS_FRONTEND})

target_link_libraries(test_onert_frontend_nnapi PRIVATE ${LIB_ONERT} dl)
target_link_libraries(test_onert_frontend_nnapi PRIVATE gtest)
target_link_libraries(test_onert_frontend_nnapi PRIVATE gtest_main)

install(TARGETS test_onert_frontend_nnapi DESTINATION unittest)
