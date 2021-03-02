file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnkit_support_backend STATIC ${SOURCES})
target_include_directories(nnkit_support_backend PUBLIC include)
target_link_libraries(nnkit_support_backend PUBLIC nnkit_intf_backend)
target_link_libraries(nnkit_support_backend PUBLIC dl)

find_package(Threads QUIET)

if(TARGET Threads::Threads)
  # This is necessary to support multi-threaded backends
  target_link_libraries(nnkit_support_backend PUBLIC Threads::Threads)
endif(TARGET Threads::Threads)
