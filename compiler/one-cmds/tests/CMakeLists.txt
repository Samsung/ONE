# Install one-cmds test scripts

# Gather test scripts
file(GLOB TESTITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.test")
file(GLOB CONFIGITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.cfg")
file(GLOB QCONFIGITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.qconf.json")
file(GLOB PYSCRIPTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.py")
file(GLOB WORKFLOWITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./*.workflow.json")

# Create a script to run the tests at installation folder
set(DRIVER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/runtestall.sh")

file(WRITE  "${DRIVER_SCRIPT}" "#!/bin/bash\n\n")
file(APPEND "${DRIVER_SCRIPT}" "SCRIPT_PATH=$(cd $(dirname $\{BASH_SOURCE\[0\]\}) && pwd)\n")
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

foreach(QCONFIGITEM IN ITEMS ${QCONFIGITEMS})
  get_filename_component(ITEM_PREFIX ${QCONFIGITEM} NAME_WE)
  install(FILES ${QCONFIGITEM} DESTINATION test)
endforeach(QCONFIGITEM)

foreach(PYSCRIPT IN ITEMS ${PYSCRIPTS})
  get_filename_component(ITEM_PREFIX ${PYSCRIPT} NAME_WE)
  install(FILES ${PYSCRIPT} DESTINATION test)
endforeach(PYSCRIPT)

foreach(WORKFLOWITEM IN ITEMS ${WORKFLOWITEMS})
  get_filename_component(ITEM_PREFIX ${WORKFLOWITEM} NAME_WE)
  install(FILES ${WORKFLOWITEM} DESTINATION test)
endforeach(WORKFLOWITEM)

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
set(PREPROCESS_IMAGES_PY "${CMAKE_CURRENT_SOURCE_DIR}/preprocess_images.py")
set(PREPROCESS_IMAGES_NUMPY_PY "${CMAKE_CURRENT_SOURCE_DIR}/preprocess_images_numpy.py")
set(ONNX_LEGALIZE_RUN_COMPARE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_legalize_run_compare.py")
set(PRINT_ONNX_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/print_onnx_model.py")

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

install(FILES ${PREPROCESS_IMAGES_PY}
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ
        DESTINATION test)

install(FILES ${PREPROCESS_IMAGES_NUMPY_PY}
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ
        DESTINATION test)
        
install(FILES ${ONNX_LEGALIZE_RUN_COMPARE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${PRINT_ONNX_MODEL}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.txt
        DESTINATION test)

add_subdirectory(onnx-operations)

if(ENABLE_ONE_IMPORT_PYTORCH)
  add_subdirectory(pytorch-operations)
endif(ENABLE_ONE_IMPORT_PYTORCH)

# Generate group option list for tests
get_filename_component(ONE_CMDS_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(ONE_PYTHON_DIR "onelib")
set(CONSTANT_EXPORTING_SCRIPT "${ONE_CMDS_DIR}/${ONE_PYTHON_DIR}/export_constant.py")
set(O1_OPTION "O1")
set(O1_TXT_FILE "${O1_OPTION}.list")
set(O1_TXT_FILE_BIN "${CMAKE_CURRENT_BINARY_DIR}/${O1_TXT_FILE}")
set(NON_O1_TXT_FILE "non-${O1_OPTION}.list")
set(NON_O1_TXT_FILE_BIN "${CMAKE_CURRENT_BINARY_DIR}/${NON_O1_TXT_FILE}")

add_custom_command(OUTPUT ${O1_TXT_FILE_BIN}
  COMMAND ${PYTHON_EXECUTABLE} ${CONSTANT_EXPORTING_SCRIPT} --constant ${O1_OPTION}
                                    --format txt --output_path ${O1_TXT_FILE_BIN}
  DEPENDS ${CONSTANT_EXPORTING_SCRIPT}
  COMMENT "Generate ${O1_TXT_FILE}"
)

add_custom_command(OUTPUT ${NON_O1_TXT_FILE_BIN}
  COMMAND ${PYTHON_EXECUTABLE} ${CONSTANT_EXPORTING_SCRIPT} --constant ${O1_OPTION}
                                    --format txt --output_path ${NON_O1_TXT_FILE_BIN}
                                    --exclusive
  DEPENDS ${CONSTANT_EXPORTING_SCRIPT}
  COMMENT "Generate ${NON_O1_TXT_FILE}"
)

add_custom_target("O1_txt_target" ALL DEPENDS ${O1_TXT_FILE_BIN} ${NON_O1_TXT_FILE_BIN})

install(FILES ${O1_TXT_FILE_BIN}
        PERMISSIONS OWNER_WRITE OWNER_READ
                         GROUP_READ
                         WORLD_READ
        DESTINATION test)

install(FILES ${NON_O1_TXT_FILE_BIN}
        PERMISSIONS OWNER_WRITE OWNER_READ
                         GROUP_READ
                         WORLD_READ
        DESTINATION test)
