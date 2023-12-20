#
# linux common compile options
#

# Disable annoying ABI compatibility warning.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  list(APPEND FLAGS_CXXONLY "-Wno-psabi")
endif()

# Build fail on memcpy (ex. compute/cker/include/cker/Shape.h:211:16)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 12.0)
  list(APPEND FLAGS_CXXONLY "-Wno-error=stringop-overflow -Wno-error=array-bounds")
endif()

# lib pthread as a variable (pthread must be disabled on android)
set(LIB_PTHREAD pthread)
