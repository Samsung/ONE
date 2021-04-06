set(SRCS_PART_TESTER
      src/Driver.cpp
      src/PModelsRunner.cpp
   )

add_executable(circle_part_driver ${SRCS_PART_TESTER})
target_link_libraries(circle_part_driver foder)
target_link_libraries(circle_part_driver loco)
target_link_libraries(circle_part_driver luci_import)
target_link_libraries(circle_part_driver luci_lang)
target_link_libraries(circle_part_driver luci_log)
target_link_libraries(circle_part_driver luci_interpreter)
target_link_libraries(circle_part_driver crew)
target_link_libraries(circle_part_driver safemain)
target_link_libraries(circle_part_driver nncc_common)

install(TARGETS circle_part_driver DESTINATION bin)
