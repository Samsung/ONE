file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(moco_pass SHARED ${SOURCES})
target_include_directories(moco_pass PRIVATE src)
target_include_directories(moco_pass PUBLIC include)
target_link_libraries(moco_pass PUBLIC loco)
target_link_libraries(moco_pass PUBLIC logo_core)
target_link_libraries(moco_pass PUBLIC moco_lang)
target_link_libraries(moco_pass PRIVATE moco_support)
target_link_libraries(moco_pass PRIVATE oops)
install(TARGETS moco_pass DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(moco_pass_test ${TESTS})
target_include_directories(moco_pass_test PRIVATE src)
target_link_libraries(moco_pass_test moco_pass)
target_link_libraries(moco_pass_test moco_support)
