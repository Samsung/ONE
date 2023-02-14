function(_Jsoncpp_import)
  nnas_find_package(JsoncppSource QUIET)

  if(NOT JsoncppSource_FOUND)
    set(Jsoncpp_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT JsoncppSource_FOUND)

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(
    CMAKE_DIR
    ${JsoncppSource_DIR}
    BUILD_DIR
    ${CMAKE_BINARY_DIR}/externals/JSONCPP/build
    INSTALL_DIR
    ${EXT_OVERLAY_DIR}
    IDENTIFIER
    "1.9.5"
    PKG_NAME
    "JSONCPP"
    EXTRA_OPTS
    "-DBUILD_STATIC_LIBS=ON"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DJSONCPP_WITH_TESTS=OFF"
    "-DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF")

  find_path(
    Jsoncpp_INCLUDE_DIRS
    NAMES json.h
    PATHS ${EXT_OVERLAY_DIR}
    NO_CMAKE_FIND_ROOT_PATH
    PATH_SUFFIXES include/json)
  find_file(
    Jsoncpp_STATIC_LIB
    NAMES libjsoncpp.a
    PATHS ${EXT_OVERLAY_DIR}
    NO_CMAKE_FIND_ROOT_PATH
    PATH_SUFFIXES lib)

  set(Jsoncpp_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_Jsoncpp_import)

_Jsoncpp_import()
