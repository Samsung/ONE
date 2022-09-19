set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(dalgona ${DRIVER} ${SOURCES})
target_include_directories(dalgona PRIVATE include)

target_link_libraries(dalgona PRIVATE arser)
target_link_libraries(dalgona PRIVATE safemain)
target_link_libraries(dalgona PRIVATE foder)
target_link_libraries(dalgona PRIVATE luci_import)
target_link_libraries(dalgona PRIVATE luci_interpreter)
target_link_libraries(dalgona PRIVATE dio_hdf5)
