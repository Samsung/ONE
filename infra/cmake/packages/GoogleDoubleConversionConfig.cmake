# https://github.com/google/double-conversion
set(GOOGLE_DOUBLE_CONVERSION_PREFIX "/usr" CACHE PATH "Google DoubleConversion install prefix")

function(_GoogleDoubleConversion_import)
  # Find the header & lib
  find_library(GoogleDoubleConversion_LIB
    NAMES double-conversion
    PATHS "${GOOGLE_DOUBLE_CONVERSION_PREFIX}/lib"
  )

  find_path(GoogleDoubleConversion_INCLUDE_DIR
    NAMES double-conversion/double-conversion.h
    PATHS "${GOOGLE_DOUBLE_CONVERSION_PREFIX}/include"
  )

  # TODO Version check
  set(GoogleDoubleConversion_FOUND TRUE)

  if(NOT GoogleDoubleConversion_LIB)
    set(GoogleDoubleConversion_FOUND FALSE)
  endif(NOT GoogleDoubleConversion_LIB)

  if(NOT GoogleDoubleConversion_INCLUDE_DIR)
    set(GoogleDoubleConversion_FOUND FALSE)
  endif(NOT GoogleDoubleConversion_INCLUDE_DIR)

  set(GoogleDoubleConversion_FOUND ${GoogleDoubleConversion_FOUND} PARENT_SCOPE)

  unset(MESSAGE)
  list(APPEND MESSAGE "Found Google Double Conversion")

  if(NOT GoogleDoubleConversion_FOUND)
    list(APPEND MESSAGE ": FALSE")
  else(NOT GoogleDoubleConversion_FOUND)
    list(APPEND MESSAGE " (include: ${GoogleDoubleConversion_INCLUDE_DIR} library: ${GoogleDoubleConversion_LIB})")

    # Add target
    if(NOT TARGET google_double_conversion)
      # NOTE IMPORTED target may be more appropriate for this case
      add_library(google_double_conversion INTERFACE)
      target_link_libraries(google_double_conversion INTERFACE ${GoogleDoubleConversion_LIB})
      target_include_directories(google_double_conversion INTERFACE ${GoogleDoubleConversion_INCLUDE_DIR})

      add_library(Google::DoubleConversion ALIAS google_double_conversion)
    endif(NOT TARGET google_double_conversion)
  endif(NOT GoogleDoubleConversion_FOUND)

  message(STATUS ${MESSAGE})
  set(GoogleDoubleConversion_FOUND ${GoogleDoubleConversion_FOUND} PARENT_SCOPE)
endfunction(_GoogleDoubleConversion_import)

_GoogleDoubleConversion_import()
