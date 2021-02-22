file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(moco_lang SHARED ${SOURCES})
target_include_directories(moco_lang PRIVATE src)
target_include_directories(moco_lang PUBLIC include)
target_link_libraries(moco_lang PUBLIC loco)
target_link_libraries(moco_lang PRIVATE nncc_common)
install(TARGETS moco_lang DESTINATION lib)  # moco_tf_frontend requires moco_lang

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(moco_lang_test ${TESTS})
target_include_directories(moco_lang_test PRIVATE src)
target_link_libraries(moco_lang_test moco_lang)
