add_executable(circle_resizer Driver.cpp)
target_link_libraries(circle_resizer PRIVATE circle_resizer_core)
target_link_libraries(circle_resizer PRIVATE arser)
target_link_libraries(circle_resizer PRIVATE safemain)

set_target_properties(circle_resizer PROPERTIES 
  OUTPUT_NAME "circle-resizer" 
)

install(TARGETS circle_resizer DESTINATION bin)
