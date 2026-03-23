file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(moco_service SHARED ${SOURCES})
target_include_directories(moco_service PRIVATE src)
target_include_directories(moco_service PUBLIC include)
target_link_libraries(moco_service PUBLIC loco)
target_link_libraries(moco_service PUBLIC moco_lang)
target_link_libraries(moco_service PRIVATE moco_support)
target_link_libraries(moco_service PRIVATE nncc_common)
target_link_libraries(moco_service PRIVATE oops)
install(TARGETS moco_service DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(moco_service_test ${TESTS})
target_include_directories(moco_service_test PRIVATE src)
target_link_libraries(moco_service_test moco_service)
