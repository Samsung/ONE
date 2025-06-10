# generate sources files by *.def files for soft backend
function(nnc_make_generated_sources DEF_SOURCES OUT_DIR GEN_SOURCES)
    set(GEN_OUT "")
    foreach(file IN LISTS DEF_SOURCES)
        get_filename_component(file_name ${file} NAME_WE)
        set(out_file "${OUT_DIR}/${file_name}.generated.h")
        list(APPEND GEN_OUT "${out_file}")
        add_custom_command(
                OUTPUT  ${out_file}
                COMMAND def2src ${OUT_DIR} ${file}
                DEPENDS def2src ${file}
        )
    endforeach()
    set(${GEN_SOURCES} ${GEN_OUT} PARENT_SCOPE)
endfunction()

function(nnc_set_installation_properties TARG)
  # TODO when we upgrade our cmake to version 3.8 we'll need to use
  #      BUILD_RPATH variable instead of CMAKE_BUILD_WITH_INSTALL_RPATH here

  # set external RPATHs
  set_target_properties(${TARG} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  # use paths from build directoris
  set_target_properties(${TARG} PROPERTIES CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  # set RPATH to core part of nnc
  set_target_properties(${TARG} PROPERTIES INSTALL_RPATH ${NNC_INSTALL_LIB_PATH})
endfunction()

# install nnc libraries
function(nnc_install_library LIB)
  install(TARGETS ${LIB} DESTINATION ${NNC_INSTALL_LIB_PATH})
  nnc_set_installation_properties(${LIB})
endfunction()

# install nnc executable
function(nnc_install_executable BIN)
  install(TARGETS ${BIN} DESTINATION ${NNC_INSTALL_BIN_PATH})
  nnc_set_installation_properties(${BIN})
endfunction()

# add nnc library as target
function(nnc_add_library)
  add_library(${ARGV})
  # to prevent _GLIBCXX17_DEPRECATED warning as error
  # target_link_libraries(${ARGV0} PRIVATE nncc_common)
  target_link_libraries(${ARGV0} PUBLIC nncc_coverage)

  get_target_property(LIBS ${NNC_TARGET_EXECUTABLE} LINK_LIBRARIES)
  target_include_directories(${ARGV0} PUBLIC ${NNC_ROOT_SRC_DIR}/include ${NNC_ROOT_BIN_DIR}/include)
  if(LIBS MATCHES NOTFOUND)
    set(LIBS "")
  endif()
  list(APPEND LIBS ${ARGV0})
  set_target_properties(${NNC_TARGET_EXECUTABLE} PROPERTIES LINK_LIBRARIES "${LIBS}")
endfunction()

# function to add nnc unit test
function(nnc_add_unit_test)
  if(ENABLE_TEST)
    add_executable(${ARGV})
    target_link_libraries(${ARGV0} gtest_main)
    add_test(${ARGV0} ${ARGV0})
  endif(ENABLE_TEST)
  add_dependencies(nnc_unit_tests ${ARGV0})
  target_include_directories(${ARGV0} PUBLIC ${NNC_ROOT_SRC_DIR}/include ${NNC_ROOT_BIN_DIR}/include)
endfunction()
