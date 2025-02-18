include(mlir/TableGen.cmake)

set(SRC
  utils/DynamicShapeUtils.cpp
)

add_library(cirmlir_dialect STATIC ${SRC})
cir_mlir_static_flags(cirmlir_dialect)
target_include_directories(cirmlir_dialect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(cirmlir_dialect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cirmlir_dialect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
