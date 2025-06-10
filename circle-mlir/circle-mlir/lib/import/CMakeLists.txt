set(SRC
  src/CircleImport.cpp
  src/CircleOperator.cpp
)

add_library(cirmlir_import STATIC ${SRC})
cir_mlir_static_flags(cirmlir_import)
target_include_directories(cirmlir_import PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(cirmlir_import PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cirmlir_import PUBLIC cirmlir_dialect)
target_link_libraries(cirmlir_import PUBLIC cirmlir_utils)
target_link_libraries(cirmlir_import PUBLIC circle_schema)
target_link_libraries(cirmlir_import PUBLIC cirmlir_coverage)
