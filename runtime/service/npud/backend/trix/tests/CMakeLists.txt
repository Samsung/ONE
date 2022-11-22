if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnfw_find_package(TRIXEngine QUIET 2.5.0)

if(NOT TRIXEngine_FOUND)
  return()
endif(NOT TRIXEngine_FOUND)

file(GLOB_RECURSE TESTS "*.test.cc")

add_executable(npud_backend_trix_test ${TESTS})

set_target_properties(npud_backend_trix_test PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_backend_trix_test PUBLIC ${NPUD_INCLUDE_DIRS})
target_include_directories(npud_backend_trix_test PUBLIC ${TRIXEngine_INCLUDE_DIRS})
target_link_libraries(npud_backend_trix_test PRIVATE npud_backend_trix)
target_link_libraries(npud_backend_trix_test PRIVATE gtest_main dl)

install(TARGETS npud_backend_trix_test DESTINATION unittest)
