function(_MbedOsSource_import)
  set(MBED_OS_SOURCE_PATH "${NNAS_EXTERNALS_DIR}/mbed-os")

  if (NOT EXISTS "${MBED_OS_SOURCE_PATH}")
    set(MBED_OS_LIB_PATH "${MBED_OS_SOURCE_PATH}.lib")

    execute_process(COMMAND ${CMAKE_COMMAND} -E echo "https://github.com/ARMmbed/mbed-os#master" > ${MBED_OS_LIB_PATH})
    execute_process(COMMAND mbed-tools deploy -f ${MBED_OS_LIB_PATH})
    file(REMOVE "${MBED_OS_LIB_PATH}")
  endif()

  set(_MbedOsSource_FOUND TRUE PARENT_SCOPE)
  set(_MbedOsSource_SOURCE_DIR "${MBED_OS_SOURCE_PATH}" PARENT_SCOPE)
endfunction()

_MbedOsSource_import()
