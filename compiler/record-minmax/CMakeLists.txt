nnas_find_package(Boost QUIET COMPONENTS program_options)
if(NOT Boost_FOUND)
  message(STATUS "Build record-minmax: FAILED (missing Boost::program_options)")
  return()
endif(NOT Boost_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(record-minmax ${DRIVER} ${SOURCES})
target_include_directories(record-minmax PRIVATE include)
target_include_directories(record-minmax PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(record-minmax ${Boost_LIBRARIES})
target_link_libraries(record-minmax safemain)
target_link_libraries(record-minmax luci_import)
target_link_libraries(record-minmax luci_export)
