#
# Platform independent compile flag setting
#
# flags for build type: debug, release
set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

#
# Platform specific compile flag setting
#
include("${CMAKE_CURRENT_LIST_DIR}/buildtool/config/config_${TARGET_PLATFORM}.cmake" OPTIONAL)

#
# Apply compile flags
# note: this should be placed after cmake/buildtool/config/config_xxx.cmake files
#
# add common flags
foreach(FLAG ${FLAGS_COMMON})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAG}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()

# add c flags
foreach(FLAG ${FLAGS_CONLY})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAG}")
endforeach()

# add cxx flags
foreach(FLAG ${FLAGS_CXXONLY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()

# lib pthread as a variable (finding pthread build option must be disabled on android)
# Define here to use on external lib build
set(LIB_PTHREAD lib_pthread)
add_library(${LIB_PTHREAD} INTERFACE)
if(NOT TARGET_OS STREQUAL "android")
  # Get compile option (ex. "-pthread" on linux GNU build tool)
  find_package(Threads)
  target_link_libraries(${LIB_PTHREAD} INTERFACE Threads::Threads)
endif()
