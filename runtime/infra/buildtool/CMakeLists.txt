cmake_minimum_required(VERSION 3.16.3)

project(nnas_buildtool)

set(ONERT_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE
  INTERNAL "Where to find nnas top-level source directory"
)
set(ONERT_EXTERNALS_DIR
  "${ONERT_PROJECT_SOURCE_DIR}/externals" CACHE
  INTERNAL "Where to download external dependencies"
)

if(NOT DEFINED BUILDTOOL_PATH)
  message(FATAL_ERROR "BUILDTOOL_PATH is not defined")
endif()

macro(nnfw_include PREFIX)
  include("${ONERT_PROJECT_SOURCE_DIR}/infra/cmake/modules/${PREFIX}.cmake")
endmacro(nnfw_include)

nnfw_include(OptionTools)
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

set(FLATBUFFERS_URL https://github.com/google/flatbuffers/archive/v24.3.25.tar.gz)

# Download and build Flatbuffers 24.3.25
include(ExternalProject)
ExternalProject_Add(flatbuffers-24.3.25
  URL ${FLATBUFFERS_URL}
  PREFIX  ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers
  SOURCE_DIR  ${ONERT_EXTERNALS_DIR}/FLATBUFFERS-HOST-24.3.25
  STAMP_DIR ${ONERT_EXTERNALS_DIR}/external-stamp
  INSTALL_DIR ${BUILDTOOL_PATH}
  CMAKE_ARGS  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${BUILDTOOL_PATH}
              -DFLATBUFFERS_BUILD_TESTS=OFF -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_BUILD_FLATHASH=OFF
)
