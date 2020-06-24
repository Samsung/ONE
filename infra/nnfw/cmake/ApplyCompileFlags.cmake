#
# Platform independent compile flag setting
#
# flags for build type: debug, release
set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE   "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

#
# Platform specific compile flag setting
#
include("${CMAKE_CURRENT_LIST_DIR}/buildtool/config/config_${TARGET_PLATFORM}.cmake")

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
