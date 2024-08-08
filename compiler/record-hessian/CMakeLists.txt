file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(record-hessian STATIC ${SOURCES})

target_include_directories(record-hessian PUBLIC include)
target_include_directories(record-hessian PRIVATE src)
target_link_libraries(record-hessian luci_import)
target_link_libraries(record-hessian luci_env)
target_link_libraries(record-hessian luci_export)
target_link_libraries(record-hessian luci_interpreter)
target_link_libraries(record-hessian luci_log)
target_link_libraries(record-hessian dio_hdf5)
install(TARGETS record-hessian DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
