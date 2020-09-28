# Install one-cmds test scripts
file(GLOB TESTITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.test")

# Create a script to run the tests at installation folder
set(DRIVER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/runtestall.sh")

file(WRITE  "${DRIVER_SCRIPT}" "#!/bin/bash\n\n")
file(APPEND "${DRIVER_SCRIPT}" "rm -rf runtestall.log\n")
file(APPEND "${DRIVER_SCRIPT}" "export PATH=${CMAKE_INSTALL_PREFIX}/bin:$PATH\n")
file(APPEND "${DRIVER_SCRIPT}" "if [[ $# -ge 1 ]]; then\n")
file(APPEND "${DRIVER_SCRIPT}" "  USER_PATH=$1\n")
file(APPEND "${DRIVER_SCRIPT}" "  export PATH=$USER_PATH:$PATH\n")
file(APPEND "${DRIVER_SCRIPT}" "fi\n")
file(APPEND "${DRIVER_SCRIPT}" "\n")

foreach(TESTITEM IN ITEMS ${TESTITEMS})
  get_filename_component(ITEM_PREFIX ${TESTITEM} NAME_WE)

  set(TESTITEM_SCRIPT_FILE "${ITEM_PREFIX}.test")
  set(TESTITEM_SCRIPT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${TESTITEM_SCRIPT_FILE}")

  file(APPEND "${DRIVER_SCRIPT}" "/bin/bash ${TESTITEM_SCRIPT_FILE} | tee -a runtestall.log\n")

  install(FILES ${TESTITEM} DESTINATION test)

endforeach(TESTITEM)

set(PREPARE_TEST_MATERIALS_SH "${CMAKE_CURRENT_SOURCE_DIR}/prepare_test_materials.sh")

install(FILES ${DRIVER_SCRIPT}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${PREPARE_TEST_MATERIALS_SH}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.txt
        DESTINATION test)
