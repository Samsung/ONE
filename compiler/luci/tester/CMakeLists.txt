set(SRCS_READ_TESTER
      src/ReadTester.cpp
      src/Model.cpp
   )

add_executable(luci_readtester "${SRCS_READ_TESTER}")
target_link_libraries(luci_readtester PRIVATE luci_import)
target_link_libraries(luci_readtester PRIVATE luci_service)
target_link_libraries(luci_readtester PRIVATE luci_pass)
target_link_libraries(luci_readtester PRIVATE oops)

set(SRCS_WRITE_TESTER
      src/WriteTester.cpp
      src/Model.cpp
   )

add_executable(luci_writetester "${SRCS_WRITE_TESTER}")
target_link_libraries(luci_writetester PRIVATE luci_import)
target_link_libraries(luci_readtester PRIVATE luci_service)
target_link_libraries(luci_writetester PRIVATE luci_pass)
target_link_libraries(luci_writetester PRIVATE luci_export)
target_link_libraries(luci_writetester PRIVATE oops)
