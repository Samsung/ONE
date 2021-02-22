nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  return()
endif(NOT Protobuf_FOUND)

# generating and building schema
Protobuf_Generate(TFINFO_PROTO
                  "${CMAKE_CURRENT_BINARY_DIR}/generated"
                  "./proto"
                  tfinfo-v2.proto)

add_library(tfinfo_v2_proto STATIC ${TFINFO_PROTO_SOURCES})
set_target_properties(tfinfo_v2_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tfinfo_v2_proto PUBLIC ${TFINFO_PROTO_INCLUDE_DIRS})
target_link_libraries(tfinfo_v2_proto PUBLIC libprotobuf)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(tfinfo_v2 STATIC ${SOURCES})
set_target_properties(tfinfo_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tfinfo_v2 PUBLIC include)
target_link_libraries(tfinfo_v2 PRIVATE tfinfo_v2_proto)
target_link_libraries(tfinfo_v2 PRIVATE oops)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(tfinfo_v2_test ${TESTS})
target_link_libraries(tfinfo_v2_test tfinfo_v2)
