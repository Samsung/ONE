#
# ExternalSource_Download(VAR ...)
#
function(ExternalSource_Download PREFIX)
  include(CMakeParseArguments)
  nnas_include(StampTools)

  cmake_parse_arguments(ARG "" "DIRNAME;URL;CHECKSUM" "" ${ARGN})

  # Configure URL
  if(ARG_URL)
    set(URL ${ARG_URL})
  else()
    # Use the first unparsed argument as URL (for backward compatibility)
    list(GET ARG_UNPARSED_ARGUMENTS 0 URL)
  endif(ARG_URL)

  # Configure DIRNAME
  if(NOT ARG_DIRNAME)
    # Use PREFIX as DIRNAME (for backward compatibility)
    set(DIRNAME ${PREFIX})
  else()
    set(DIRNAME ${ARG_DIRNAME})
  endif(NOT ARG_DIRNAME)

  get_filename_component(FILENAME ${URL} NAME)

  set(CACHE_DIR "${NNAS_EXTERNALS_DIR}")
  set(OUT_DIR "${CACHE_DIR}/${DIRNAME}")
  set(TMP_DIR "${CACHE_DIR}/${DIRNAME}-tmp")

  set(DOWNLOAD_PATH "${CACHE_DIR}/${DIRNAME}-${FILENAME}")
  set(STAMP_PATH "${CACHE_DIR}/${DIRNAME}.stamp")

  if(NOT EXISTS "${CACHE_DIR}")
    file(MAKE_DIRECTORY "${CACHE_DIR}")
  endif(NOT EXISTS "${CACHE_DIR}")

  # Compare URL in STAMP file and the given URL
  Stamp_Check(URL_CHECK "${STAMP_PATH}" "${URL}")

  if(NOT EXISTS "${OUT_DIR}" OR NOT URL_CHECK)
    file(REMOVE "${STAMP_PATH}")
    file(REMOVE_RECURSE "${OUT_DIR}")
    file(REMOVE_RECURSE "${TMP_DIR}")

    file(MAKE_DIRECTORY "${TMP_DIR}")

    message(STATUS "Download ${PREFIX} from ${URL}")

    foreach(retry_count RANGE 5)
      message(STATUS "(Trial Count : ${retry_count})")

      file(DOWNLOAD ${URL} "${DOWNLOAD_PATH}"
                    STATUS status
                    LOG log)

      list(GET status 0 status_code)
      list(GET status 1 status_string)

      # Download success
      if(status_code EQUAL 0)
        break()
      endif()

      message(WARNING "error: downloading '${URL}' failed
              status_code: ${status_code}
              status_string: ${status_string}
              log: ${log}")

      # Retry limit exceed
      if(retry_count EQUAL 5)
        message(FATAL_ERROR "Download ${PREFIX} from ${URL} - failed")
      endif()
      
      # Retry after 10 seconds when download fails
      execute_process(COMMAND sleep 10)
    endforeach()

    message(STATUS "Download ${PREFIX} from ${URL} - done")

    # Verify checksum
    if(ARG_CHECKSUM)
      message(STATUS "Verify ${PREFIX} archive")
      string(REPLACE "=" ";" CHECKSUM_SPEC "${ARG_CHECKSUM}")

      list(GET CHECKSUM_SPEC 0 CHECKSUM_ALG)
      list(GET CHECKSUM_SPEC 1 CHECKSUM_VAL)
      string(STRIP "${CHECKSUM_VAL}" CHECKSUM_VAL)

      set(EXPECTED_CHECKSUM ${CHECKSUM_VAL})
      file(${CHECKSUM_ALG} "${DOWNLOAD_PATH}" OBTAINED_CHECKSUM)

      if(NOT (EXPECTED_CHECKSUM STREQUAL OBTAINED_CHECKSUM))
        message(STATUS "CHECKSUM MISMATCH")
        message(STATUS "  expected: ${EXPECTED_CHECKSUM}")
        message(STATUS "  obtained: ${OBTAINED_CHECKSUM}")
        message(FATAL_ERROR "Verify ${PREFIX} archive - fail")
      endif(NOT (EXPECTED_CHECKSUM STREQUAL OBTAINED_CHECKSUM))

      message(STATUS "Verify ${PREFIX} archive - done")
    endif(ARG_CHECKSUM)

    message(STATUS "Extract ${PREFIX}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz "${DOWNLOAD_PATH}"
                    WORKING_DIRECTORY "${TMP_DIR}"
                    ERROR_VARIABLE EXTRACTION_ERROR)

    if(EXTRACTION_ERROR)
      message(FATAL_ERROR "Extract ${PREFIX} - failed")
    endif(EXTRACTION_ERROR)

    file(REMOVE "${DOWNLOAD_PATH}")
    message(STATUS "Extract ${PREFIX} - done")

    message(STATUS "Cleanup ${PREFIX}")
    file(GLOB contents "${TMP_DIR}/*")
    list(LENGTH contents n)
    if(NOT n EQUAL 1 OR NOT IS_DIRECTORY "${contents}")
      set(contents "${TMP_DIR}")
    endif()

    get_filename_component(contents ${contents} ABSOLUTE)

    file(RENAME ${contents} "${OUT_DIR}")
    file(REMOVE_RECURSE "${TMP_DIR}")
    file(WRITE "${STAMP_PATH}" "${URL}")
    message(STATUS "Cleanup ${PREFIX} - done")
  endif()

  set(${PREFIX}_SOURCE_DIR "${OUT_DIR}" PARENT_SCOPE)
endfunction(ExternalSource_Download)
