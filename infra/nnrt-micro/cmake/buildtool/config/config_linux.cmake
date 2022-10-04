#
# linux common compile options
#

# Disable annoying ABI compatibility warning.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  list(APPEND FLAGS_CXXONLY "-Wno-psabi")
endif()

# lib pthread as a variable (pthread must be disabled on android)
set(LIB_PTHREAD pthread)
