file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(logo_ex STATIC ${SOURCES})
if(NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(logo_ex PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)
target_include_directories(logo_ex PRIVATE src)
target_include_directories(logo_ex PUBLIC include)
target_link_libraries(logo_ex PUBLIC loco)
target_link_libraries(logo_ex PUBLIC logo_core)
target_link_libraries(logo_ex PRIVATE locomotiv)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(logo_ex_test ${TESTS})
target_include_directories(logo_ex_test PRIVATE src)
target_link_libraries(logo_ex_test logo_ex)
