nnas_find_package(Jsoncpp)
if(NOT Jsoncpp_FOUND)
  message(STATUS "Build jsoncpp: FAILED (missing jsoncpp)")
  return()
endif(NOT Jsoncpp_FOUND)

set (SOURCES src/LibCircleQuantizer.cpp)

add_library(circle-quantizer "${SOURCES}")
target_include_directories(circle-quantizer PRIVATE ${Jsoncpp_INCLUDE_DIRS})

target_link_libraries(circle-quantizer ${Jsoncpp_STATIC_LIB})
target_link_libraries(circle-quantizer oops)
target_link_libraries(circle-quantizer loco)
target_link_libraries(circle-quantizer luci_import)
target_link_libraries(circle-quantizer luci_service)
target_link_libraries(circle-quantizer luci_pass)
target_link_libraries(circle-quantizer luci_export)
target_link_libraries(circle-quantizer luci_env)
target_link_libraries(circle-quantizer arser)
target_link_libraries(circle-quantizer vconone)

install(TARGETS circle-quantizer DESTINATION lib)
