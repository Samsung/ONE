# Install onecc-docker test scripts

# Gather test scripts
file(GLOB TESTITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.test")
file(GLOB CONFIGITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.cfg")

# Create a script to run the tests at installation folder
set(DRIVER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/runtestall.sh")

file(WRITE  "${DRIVER_SCRIPT}" "#!/bin/bash\n\n")
file(APPEND "${DRIVER_SCRIPT}" "SCRIPT_PATH=$(cd $(dirname $\{BASH_SOURCE\[0\]\}) && pwd)")
file(APPEND "${DRIVER_SCRIPT}" "\n")
file(APPEND "${DRIVER_SCRIPT}" "pushd $SCRIPT_PATH > /dev/null\n")
file(APPEND "${DRIVER_SCRIPT}" "rm -rf runtestall.log\n")
file(APPEND "${DRIVER_SCRIPT}" "export PATH=$SCRIPT_PATH/../bin:$PATH\n")
file(APPEND "${DRIVER_SCRIPT}" "if [[ $# -ge 1 ]]; then\n")
file(APPEND "${DRIVER_SCRIPT}" "  USER_PATH=$1\n")
file(APPEND "${DRIVER_SCRIPT}" "  export PATH=$USER_PATH:$PATH\n")
file(APPEND "${DRIVER_SCRIPT}" "fi\n")
file(APPEND "${DRIVER_SCRIPT}" "\n")
file(APPEND "${DRIVER_SCRIPT}" "# refer https://github.com/Samsung/ONE/issues/6286\n")
file(APPEND "${DRIVER_SCRIPT}" "set -o pipefail\n\n")
file(APPEND "${DRIVER_SCRIPT}" "fail_count=0\n")
file(APPEND "${DRIVER_SCRIPT}" "trap \"(( fail_count++ ))\" ERR\n\n")

foreach(TESTITEM IN ITEMS ${TESTITEMS})
  get_filename_component(ITEM_PREFIX ${TESTITEM} NAME_WE)

  set(TESTITEM_SCRIPT_FILE "${ITEM_PREFIX}.test")
  set(TESTITEM_SCRIPT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${TESTITEM_SCRIPT_FILE}")

  file(APPEND "${DRIVER_SCRIPT}" "/bin/bash ${TESTITEM_SCRIPT_FILE} | tee -a runtestall.log\n")

  install(FILES ${TESTITEM} DESTINATION test)
endforeach(TESTITEM)

foreach(CONFIGITEM IN ITEMS ${CONFIGITEMS})
  get_filename_component(ITEM_PREFIX ${CONFIGITEM} NAME_WE)
  install(FILES ${CONFIGITEM} DESTINATION test)
endforeach(CONFIGITEM)

file(APPEND "${DRIVER_SCRIPT}" "popd > /dev/null\n\n")

file(APPEND "${DRIVER_SCRIPT}"
"if [[ $fail_count != 0 ]]; then
  echo \"$fail_count TESTS FAILED\"
  exit 255
else
  echo \"ALL TESTS PASSED!\"
fi\n
")

set(PREPARE_TEST_MATERIALS_SH "${CMAKE_CURRENT_SOURCE_DIR}/prepare_test_materials.sh")

install(FILES ${DRIVER_SCRIPT}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test/onecc-docker)

install(FILES ${PREPARE_TEST_MATERIALS_SH}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test/onecc-docker)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.txt
        DESTINATION test/onecc-docker)
