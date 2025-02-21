set(SRC
  src/CircleExport.cpp
  src/OpOrArgNameMapper.cpp
)

add_library(cirmlir_export STATIC ${SRC})
cir_mlir_static_flags(cirmlir_export)
target_include_directories(cirmlir_export PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(cirmlir_export PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cirmlir_export PUBLIC cirmlir_dialect)
target_link_libraries(cirmlir_export PUBLIC cirmlir_utils)
target_link_libraries(cirmlir_export PUBLIC circle_schema)
target_link_libraries(cirmlir_export PUBLIC abseil_cpp)
target_link_libraries(cirmlir_export PUBLIC cirmlir_coverage)
