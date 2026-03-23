file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(moco_import SHARED ${SOURCES})
target_include_directories(moco_import PRIVATE src)
target_include_directories(moco_import PUBLIC include)
target_link_libraries(moco_import PUBLIC moco_lang)
target_link_libraries(moco_import PUBLIC mio_tf)
target_link_libraries(moco_import PRIVATE nncc_common)
target_link_libraries(moco_import PRIVATE plier_tf)
target_link_libraries(moco_import PRIVATE oops)
install(TARGETS moco_import DESTINATION lib)  # moco_tf_frontend requires moco_import

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(moco_import_test ${TESTS})
target_include_directories(moco_import_test PRIVATE src)
target_link_libraries(moco_import_test moco_import)
target_link_libraries(moco_import_test plier_tf)
target_link_libraries(moco_import_test oops)
