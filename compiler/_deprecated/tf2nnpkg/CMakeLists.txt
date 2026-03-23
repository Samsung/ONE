if(NOT TARGET moco_tf_frontend)
  return()
endif(NOT TARGET moco_tf_frontend)

if(NOT TARGET tfinfo)
  return()
endif(NOT TARGET tfinfo)

if(NOT TARGET exo)
  return()
endif(NOT TARGET exo)

message(STATUS "Build tf2nnpkg: TRUE")

unset(SOURCES)
list(APPEND SOURCES "src/tf2nnpkg.cpp")
list(APPEND SOURCES "src/filesystem_common.cpp")

option(TF2NNPKG_FOR_WINDOWS "Use Windows filesystem" OFF)
if(TF2NNPKG_FOR_WINDOWS)
  list(APPEND SOURCES "src/filesystem_windows.cpp")
else()
  list(APPEND SOURCES "src/filesystem_linux.cpp")  # default
endif()

add_executable(tf2nnpkg ${SOURCES})
target_link_libraries(tf2nnpkg PRIVATE moco_log)
target_link_libraries(tf2nnpkg PRIVATE moco_tf_frontend)
target_link_libraries(tf2nnpkg PRIVATE tfinfo)
target_link_libraries(tf2nnpkg PRIVATE exo)
target_link_libraries(tf2nnpkg PRIVATE locop)
target_link_libraries(tf2nnpkg PRIVATE hermes_std)
target_link_libraries(tf2nnpkg PRIVATE angkor cwrap)
install(TARGETS tf2nnpkg DESTINATION bin)
