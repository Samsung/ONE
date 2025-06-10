set(SRC
  src/ConvertType.cpp
  src/FlatbufferOperator.cpp
  src/SizeUtils.cpp
)

add_library(cirmlir_utils STATIC ${SRC})
cir_mlir_static_flags(cirmlir_utils)
target_include_directories(cirmlir_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(cirmlir_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cirmlir_utils PUBLIC cirmlir_dialect)
target_link_libraries(cirmlir_utils PUBLIC circle_schema)
target_link_libraries(cirmlir_utils PUBLIC cirmlir_coverage)

if(NOT ENABLE_TEST)
  return()
endif()

set(SRC_TEST
  src/ConvertType.test.cpp
  src/SizeUtils.test.cpp
)

GTest_AddTest(cirmlir_utils_test ${SRC_TEST})
target_link_libraries(cirmlir_utils_test cirmlir_utils)
