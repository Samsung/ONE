set(SRC
  src/CirclePass.cpp
  src/ConvertONNXToCirclePass.cpp
  src/ConvertHelper.cpp
  src/RewriteCirclePass.cpp
  src/RewriteONNXPass.cpp
  src/DumpCircleOpsPass.cpp
  src/RuntimeVerifyPass.cpp
  src/ShapeInferencePass.cpp
)

add_library(cirmlir_pass STATIC ${SRC})
cir_mlir_static_flags(cirmlir_pass)
cir_onnx_static_flags(cirmlir_pass)
target_include_directories(cirmlir_pass PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(cirmlir_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cirmlir_pass PUBLIC cirmlir_dialect)
target_link_libraries(cirmlir_pass PUBLIC cirmlir_coverage)

add_dependencies(cirmlir_pass circle_mlir_gen_inc)
target_include_directories(cirmlir_pass PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
