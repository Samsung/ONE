set(DRIVER "driver/Driver.cpp")

add_executable(dalgona ${DRIVER})

target_link_libraries(dalgona PRIVATE arser)
target_link_libraries(dalgona PRIVATE safemain)
