# TODO Allow users to force tf2circle build
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
Protobuf_Generate(CIRCLE_CUSTOMOP_INFO_PROTO
                  "${CMAKE_CURRENT_BINARY_DIR}/generated"
                  "./proto"
                  CustomOpInfo.proto)

add_library(tf2circle_customop_info_proto STATIC ${CIRCLE_CUSTOMOP_INFO_PROTO_SOURCES})
set_target_properties(tf2circle_customop_info_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tf2circle_customop_info_proto PUBLIC ${CIRCLE_CUSTOMOP_INFO_PROTO_INCLUDE_DIRS})
target_link_libraries(tf2circle_customop_info_proto PUBLIC libprotobuf)
install(TARGETS tf2circle_customop_info_proto DESTINATION lib)

message(STATUS "Build tf2circle: TRUE")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tf2circle ${SOURCES})
target_link_libraries(tf2circle PRIVATE moco_log)
target_link_libraries(tf2circle PRIVATE moco_tf_frontend)
target_link_libraries(tf2circle PRIVATE tfinfo)
target_link_libraries(tf2circle PRIVATE exo)
target_link_libraries(tf2circle PRIVATE locop)
target_link_libraries(tf2circle PRIVATE hermes_std)
target_link_libraries(tf2circle PRIVATE angkor cwrap)
target_link_libraries(tf2circle PRIVATE tf2circle_customop_info_proto)

install(TARGETS tf2circle DESTINATION bin)
