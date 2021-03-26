set(SRCS_PART_TESTER
      src/Driver.cpp
   )

add_executable(circle_part_driver ${SRCS_PART_TESTER})
target_link_libraries(circle_part_driver safemain)

install(TARGETS circle_part_driver DESTINATION bin)
