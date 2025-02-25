include(mlir/TableGen.cmake)

set(SRC
  src/CircleDialect.cpp
  src/NameUtils.cpp
  src/ShapeInference.cpp
  utils/DynamicShapeUtils.cpp
  utils/Errors.cpp
  utils/KernelShapeUtil.cpp
  utils/Padding.cpp
)

add_library(cirmlir_dialect STATIC ${SRC})
cir_mlir_static_flags(cirmlir_dialect)
target_include_directories(cirmlir_dialect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(cirmlir_dialect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cirmlir_dialect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# use generated files
add_dependencies(cirmlir_dialect circle_mlir_gen_inc)
target_include_directories(cirmlir_dialect PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cirmlir_dialect PUBLIC abseil_cpp)
target_link_libraries(cirmlir_dialect PUBLIC circle_schema)
target_link_libraries(cirmlir_dialect PUBLIC cirmlir_coverage)

if(NOT ENABLE_TEST)
  return()
endif()

set(SRC_TEST
  utils/KernelShapeUtil.test.cpp
  utils/Padding.test.cpp
)

GTest_AddTest(cirmlir_dialect_test ${SRC_TEST})
target_link_libraries(cirmlir_dialect_test cirmlir_dialect)
