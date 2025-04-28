cmake_minimum_required(VERSION 3.16.3)

project(nnas_buildtool)

set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE
  INTERNAL "Where to find nnas top-level source directory"
)
set(NNAS_EXTERNALS_DIR
  "${NNAS_PROJECT_SOURCE_DIR}/externals" CACHE
  INTERNAL "Where to download external dependencies"
)

if(NOT DEFINED BUILDTOOL_PATH)
  message(FATAL_ERROR "BUILDTOOL_PATH is not defined")
endif()

macro(nnas_include PREFIX)
  include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/${PREFIX}.cmake")
endmacro(nnas_include)

nnas_include(OptionTools)
envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
# Sync with traditional external download tool: server login
# TODO Migrate external download tool
envoption(EXTERNAL_SERVER_USERPWD "")
if(EXTERNAL_SERVER_USERPWD)
  string(REGEX MATCH "^[^:]*" EXTERNAL_SERVER_USERNAME ${EXTERNAL_SERVER_USERPWD})
  string(REGEX MATCH "[^:]*$" EXTERNAL_SERVER_PASSWORD ${EXTERNAL_SERVER_USERPWD})
else()
  envoption(EXTERNAL_SERVER_USERNAME "")
  envoption(EXTERNAL_SERVER_PASSWORD "")
endif()

set(FLATBUFFERS_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/flatbuffers/archive/v23.5.26.tar.gz)

# Download and build Flatbuffers 23.5.26
include(ExternalProject)
ExternalProject_Add(flatbuffers-23.5.26
  URL ${FLATBUFFERS_URL}
  HTTP_USERNAME ${EXTERNAL_SERVER_USERNAME}
  HTTP_PASSWORD ${EXTERNAL_SERVER_PASSWORD}
  PREFIX  ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers
  SOURCE_DIR  ${NNAS_EXTERNALS_DIR}/FLATBUFFERS-23.5.26
  STAMP_DIR ${NNAS_EXTERNALS_DIR}/external-stamp
  INSTALL_DIR ${BUILDTOOL_PATH}
  CMAKE_ARGS  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${BUILDTOOL_PATH}
              -DFLATBUFFERS_BUILD_TESTS=OFF -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_BUILD_FLATHASH=OFF
)

# Sync with traditional external download tool - stamp
# TODO Migrate external download tool
nnas_include(StampTools)
set(STAMP_PATH "${NNAS_EXTERNALS_DIR}/FLATBUFFERS-23.5.26.stamp")
Stamp_Check(URL_CHECK "${STAMP_PATH}" "${FLATBUFFERS_URL}")
if(NOT URL_CHECK)
  file(WRITE "${STAMP_PATH}" "${FLATBUFFERS_URL}")
endif(NOT URL_CHECK)
