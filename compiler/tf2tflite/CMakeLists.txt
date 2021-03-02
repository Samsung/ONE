# TODO Allow users to force tf2tflite build
if(NOT TARGET moco_tf_frontend)
  return()
endif(NOT TARGET moco_tf_frontend)

if(NOT TARGET tfinfo)
  return()
endif(NOT TARGET tfinfo)

if(NOT TARGET exo)
  return()
endif(NOT TARGET exo)

nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  return()
endif(NOT Protobuf_FOUND)

# generating and building schema for customop.conf
Protobuf_Generate(CUSTOMOP_INFO_PROTO
                  "${CMAKE_CURRENT_BINARY_DIR}/generated"
                  "./proto"
                  CustomOpInfo.proto)

add_library(tf2tflite_customop_info_proto SHARED ${CUSTOMOP_INFO_PROTO_SOURCES})
set_target_properties(tf2tflite_customop_info_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tf2tflite_customop_info_proto PUBLIC ${CUSTOMOP_INFO_PROTO_INCLUDE_DIRS})
target_link_libraries(tf2tflite_customop_info_proto PUBLIC libprotobuf)
install(TARGETS tf2tflite_customop_info_proto DESTINATION lib)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tf2tflite ${SOURCES})
target_link_libraries(tf2tflite PRIVATE moco_log)
target_link_libraries(tf2tflite PRIVATE moco_tf_frontend)
target_link_libraries(tf2tflite PRIVATE tfinfo)
target_link_libraries(tf2tflite PRIVATE exo)
target_link_libraries(tf2tflite PRIVATE locop)
target_link_libraries(tf2tflite PRIVATE hermes_std)
target_link_libraries(tf2tflite PRIVATE angkor cwrap)
target_link_libraries(tf2tflite PRIVATE tf2tflite_customop_info_proto)
install(TARGETS tf2tflite DESTINATION bin)
