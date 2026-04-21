include(UseMLIR)
include(UseAbseil)

add_subdirectory(lib)
add_subdirectory(tools)
if(ENABLE_TEST)
  add_subdirectory(tools-test)
endif()
