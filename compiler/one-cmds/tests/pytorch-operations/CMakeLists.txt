# Install one-cmds test scripts for pytorch models

# Gather test scripts
set(EXAMPLES_DIR "${NNAS_PROJECT_SOURCE_DIR}/res/PyTorchExamples/examples")
file(GLOB TEST_EXAMPLES RELATIVE "${EXAMPLES_DIR}" "${EXAMPLES_DIR}/*")

set(TEST_DST test/pytorch-operations)

install(DIRECTORY "${NNAS_PROJECT_SOURCE_DIR}/res/PyTorchExamples/" DESTINATION "${TEST_DST}")

foreach(TEST_ITEM IN ITEMS ${TEST_EXAMPLES})
  set(TEST_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TEST_ITEM}.test")

  # generate test script
  file(WRITE  "${TEST_SCRIPT}" "#!/bin/bash\n\n")
  file(APPEND "${TEST_SCRIPT}" "filename_ext=\"\$(basename -- $0)\"\n")
  file(APPEND "${TEST_SCRIPT}" "filename=\"\${filename_ext%.*}\"\n")
  file(APPEND "${TEST_SCRIPT}" "trap_err_onexit()\n")
  file(APPEND "${TEST_SCRIPT}" "{\n")
  file(APPEND "${TEST_SCRIPT}" "  echo \"\${filename_ext} FAILED\"\n")
  file(APPEND "${TEST_SCRIPT}" "  exit 255\n")
  file(APPEND "${TEST_SCRIPT}" "}\n")
  file(APPEND "${TEST_SCRIPT}" "trap trap_err_onexit ERR\n")
  file(APPEND "${TEST_SCRIPT}" "outputfile=\"${TEST_ITEM}.circle\"\n")
  file(APPEND "${TEST_SCRIPT}" "input_shapes=\$(head -n 1 ${TEST_ITEM}.spec)\n")
  file(APPEND "${TEST_SCRIPT}" "input_types=\$(tail -n 1 ${TEST_ITEM}.spec)\n")
  file(APPEND "${TEST_SCRIPT}" "one-import-pytorch --input_path=${TEST_ITEM}.pth --output_path=${TEST_ITEM}.circle --input_shapes=\${input_shapes} --input_types=\${input_types} &> /dev/null\n")
  file(APPEND "${TEST_SCRIPT}" "if [[ ! -s \"\${outputfile}\" ]]; then\n")
  file(APPEND "${TEST_SCRIPT}" "  trap_err_onexit\n")
  file(APPEND "${TEST_SCRIPT}" "fi\n")
  file(APPEND "${TEST_SCRIPT}" "echo \"\${filename_ext} SUCCESS\"\n")

  install(FILES "${TEST_SCRIPT}" DESTINATION "${TEST_DST}")
endforeach(TEST_ITEM)


# Create a script to run the tests at installation folder
set(DRIVER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/runtestall.sh")

file(WRITE  "${DRIVER_SCRIPT}" "#!/bin/bash\n\n")
file(APPEND "${DRIVER_SCRIPT}" "SCRIPT_PATH=$(cd $(dirname \${BASH_SOURCE[0]}) && pwd)\n")
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

foreach(TEST_ITEM IN ITEMS ${TEST_EXAMPLES})
  file(APPEND "${DRIVER_SCRIPT}" "/bin/bash \"${TEST_ITEM}.test\" | tee -a runtestall.log\n")
endforeach(TEST_ITEM)

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
set(EXAMPLE_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/example_generator.py")
set(AUX_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/aux_generator.py")

install(FILES "${DRIVER_SCRIPT}"
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION "${TEST_DST}")

install(FILES "${PREPARE_TEST_MATERIALS_SH}"
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION "${TEST_DST}")

install(FILES "${EXAMPLE_GENERATOR}" "${AUX_GENERATOR}"
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
        DESTINATION "${TEST_DST}")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        DESTINATION "${TEST_DST}")
