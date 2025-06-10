function(ThirdParty_URL VARNAME)
  # PACKAGE (mandatory)
  # VERSION (mandatory)
  # ENV ... (optional, for backward compatibility)

  include(CMakeParseArguments)

  cmake_parse_arguments(ARG "" "PACKAGE;VERSION;ENV" "" ${ARGN})

  if(NOT ARG_PACKAGE)
    message(FATAL_ERROR "PACKAGE is missing")
  endif(NOT ARG_PACKAGE)

  if(NOT ARG_VERSION)
    message(FATAL_ERROR "VERSION is missing")
  endif(NOT ARG_VERSION)

  set(PACKAGE_INFO_DIR "${NNAS_PROJECT_SOURCE_DIR}/infra/3rdparty/${ARG_PACKAGE}/${ARG_VERSION}")
  set(PACKAGE_URL_FILE "${PACKAGE_INFO_DIR}/URL.default")
  set(PACKAGE_URL_LOCAL_FILE "${PACKAGE_INFO_DIR}/URL.local")

  if(NOT EXISTS "${PACKAGE_URL_FILE}")
    message(FATAL_ERROR "URL file does not exist")
  endif()

  # Read URL from "[PACKAGE NAME]/[PACKAGE VERSION]/URL.default"
  file(STRINGS "${PACKAGE_URL_FILE}" VALUE)

  # Read URL from "[PACKAGE NAME]/[PACAKGE VERSION]/URL.local" (if it exists)
  if(EXISTS "${PACKAGE_URL_LOCAL_FILE}")
    file(STRINGS "${PACKAGE_URL_LOCAL_FILE}" VALUE)
  endif()

  # Read URL from process environment (if ENV option is specified)
  if(ARG_ENV)
    if(DEFINED ENV{${ARG_ENV}})
      set(VALUE $ENV{${ARG_ENV}})
    endif()
  endif(ARG_ENV)

  set("${VARNAME}" "${VALUE}" PARENT_SCOPE)
endfunction(ThirdParty_URL)
