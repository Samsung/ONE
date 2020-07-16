add_executable(circlechef-file Driver.cpp)
target_link_libraries(circlechef-file arser)
target_link_libraries(circlechef-file circlechef_core)
target_link_libraries(circlechef-file safemain)
