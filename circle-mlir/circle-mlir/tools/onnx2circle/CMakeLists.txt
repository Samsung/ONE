set(SRC
  src/onnx2circle.cpp
)

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug")
  list(APPEND SRC src/driverDebug.cpp)
else()
  list(APPEND SRC src/driverRelease.cpp)
  set(RELEASE_BUILD ON)
endif()

add_executable(onnx2circle ${SRC})
cir_mlir_static_flags(onnx2circle)
cir_onnx_static_flags(onnx2circle)
cir_onnx_tools_flags(onnx2circle)
target_link_libraries(onnx2circle PUBLIC cirmlir_dialect)
target_link_libraries(onnx2circle PUBLIC cirmlir_pass)
target_link_libraries(onnx2circle PUBLIC cirmlir_export)
target_link_libraries(onnx2circle PUBLIC cirmlir_coverage)
if(RELEASE_BUILD)
  target_link_libraries(onnx2circle PUBLIC arser)
endif()

install(TARGETS onnx2circle DESTINATION bin)
