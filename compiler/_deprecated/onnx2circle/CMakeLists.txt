# TODO Allow users to force onnx2circle build
if(NOT TARGET mir2loco)
  return()
endif(NOT TARGET mir2loco)

if(NOT TARGET mir_onnx_importer)
  return()
endif(NOT TARGET mir_onnx_importer)

if(NOT TARGET exo)
  return()
endif(NOT TARGET exo)

message(STATUS "Build onnx2circle: TRUE")

set(SOURCES "src/onnx2circle.cpp")

add_executable(onnx2circle ${SOURCES})
target_link_libraries(onnx2circle PRIVATE moco_log)
target_link_libraries(onnx2circle PRIVATE exo)
target_link_libraries(onnx2circle PRIVATE locop)
target_link_libraries(onnx2circle PRIVATE hermes_std)
target_link_libraries(onnx2circle PRIVATE angkor cwrap)
target_link_libraries(onnx2circle PRIVATE mir2loco)
target_link_libraries(onnx2circle PRIVATE mir_onnx_importer)
install(TARGETS onnx2circle DESTINATION bin)
