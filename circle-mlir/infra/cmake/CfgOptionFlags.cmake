#
# configuration options
#
option(ENABLE_TEST "Enable module unit test and intrgration test" ON)
option(ENABLE_COVERAGE "Enable build for coverage test" OFF)

if(${ENABLE_COVERAGE} AND NOT ${ENABLE_TEST})
  message(FATAL_ERROR "Test should be enabled to measure test coverage")
endif()
