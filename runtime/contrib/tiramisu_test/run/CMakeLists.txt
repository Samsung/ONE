SET(OBJS ${CMAKE_CURRENT_SOURCE_DIR}/arithmetic.o)
SET(SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)

add_executable(tiramisu_test ${OBJS} ${SRCS})

target_include_directories(tiramisu_test PRIVATE Halide/include)
target_link_libraries(tiramisu_test PRIVATE pthread dl)

set_source_files_properties(
  ${OBJS}
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true
)
