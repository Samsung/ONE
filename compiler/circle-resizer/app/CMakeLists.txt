add_executable(circle_resizer CircleResizer.cpp)
target_link_libraries(circle_resizer PRIVATE circle_resizer_core)
target_link_libraries(circle_resizer PRIVATE arser)
target_link_libraries(circle_resizer PRIVATE safemain)
target_link_libraries(circle_resizer PRIVATE vconone)

set_target_properties(circle_resizer PROPERTIES
  OUTPUT_NAME "circle-resizer"
)

install(TARGETS circle_resizer DESTINATION bin)
