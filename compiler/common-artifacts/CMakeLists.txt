#[[ Generate common python virtual enviornment ]]
find_package(PythonInterp 3 QUIET)
find_package(PythonLibs 3 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build common-artifacts: FALSE (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 3)
  message(STATUS "Build common-artifacts: FALSE (You need to install Python version higher than 3.3)")
  return()
endif()

# Create python virtual environment with tensorflow 1.13.2
set(VIRTUALENV_OVERLAY_TF_1_13_2 "${NNCC_OVERLAY_DIR}/venv_1_13_2")

# Create python virtual environment with tensorflow 2.3.0
set(VIRTUALENV_OVERLAY_TF_2_3_0 "${NNCC_OVERLAY_DIR}/venv_2_3_0")

add_custom_command(
  OUTPUT ${VIRTUALENV_OVERLAY_TF_1_13_2}
  COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV_OVERLAY_TF_1_13_2}
)

add_custom_command(
  OUTPUT ${VIRTUALENV_OVERLAY_TF_2_3_0}
  COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV_OVERLAY_TF_2_3_0}
)

# Create requirements.txt and install required pip packages
set(REQUIREMENTS_FILE "requirements.txt")
set(REQUIREMENTS_OVERLAY_PATH_TF_1_13_2 "${VIRTUALENV_OVERLAY_TF_1_13_2}/${REQUIREMENTS_FILE}")
set(REQUIREMENTS_OVERLAY_PATH_TF_2_3_0 "${VIRTUALENV_OVERLAY_TF_2_3_0}/${REQUIREMENTS_FILE}")

add_custom_command(
  OUTPUT ${REQUIREMENTS_OVERLAY_PATH_TF_1_13_2}
  COMMAND ${CMAKE_COMMAND} -E echo "tensorflow==1.13.2" > ${REQUIREMENTS_OVERLAY_PATH_TF_1_13_2}
  COMMAND ${VIRTUALENV_OVERLAY_TF_1_13_2}/bin/python -m pip --default-timeout=1000 install --upgrade pip setuptools
  COMMAND ${VIRTUALENV_OVERLAY_TF_1_13_2}/bin/python -m pip --default-timeout=1000 install -r ${REQUIREMENTS_OVERLAY_PATH_TF_1_13_2} --upgrade
  DEPENDS ${VIRTUALENV_OVERLAY_TF_1_13_2}
)

add_custom_command(
  OUTPUT ${REQUIREMENTS_OVERLAY_PATH_TF_2_3_0}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${REQUIREMENTS_OVERLAY_PATH_TF_2_3_0}
  COMMAND ${CMAKE_COMMAND} -E echo "tensorflow-cpu==2.3.0" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_3_0}
  COMMAND ${CMAKE_COMMAND} -E echo "flatbuffers==1.12" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_3_0}
  COMMAND ${VIRTUALENV_OVERLAY_TF_2_3_0}/bin/python -m pip --default-timeout=1000 install --upgrade pip setuptools
  COMMAND ${VIRTUALENV_OVERLAY_TF_2_3_0}/bin/python -m pip --default-timeout=1000 install -r ${REQUIREMENTS_OVERLAY_PATH_TF_2_3_0} --upgrade
  DEPENDS ${VIRTUALENV_OVERLAY_TF_2_3_0}
)

add_custom_target(common_artifacts_python_deps ALL
  DEPENDS ${VIRTUALENV_OVERLAY_TF_1_13_2} ${VIRTUALENV_OVERLAY_TF_2_3_0} ${REQUIREMENTS_OVERLAY_PATH_TF_1_13_2} ${REQUIREMENTS_OVERLAY_PATH_TF_2_3_0}
)

#[[ Generate common resources ]]
# TODO add pbtxt
nnas_find_package(HDF5 QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build common-artifacts: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

set(SOURCES src/TestDataGenerator.cpp)

add_executable(testDataGenerator ${SOURCES})
target_include_directories(testDataGenerator PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(testDataGenerator PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(testDataGenerator PRIVATE arser)
target_link_libraries(testDataGenerator PRIVATE foder)
target_link_libraries(testDataGenerator PRIVATE luci_import)
target_link_libraries(testDataGenerator PRIVATE luci_interpreter)
target_link_libraries(testDataGenerator PRIVATE mio_circle)
target_link_libraries(testDataGenerator PRIVATE safemain)

unset(TEST_DEPS)

# Include recipe repo
nncc_find_resource(TensorFlowLiteRecipes)
nncc_find_resource(CircleRecipes)
set(TFLITE_RECIPE_REPO "${TensorFlowLiteRecipes_DIR}")
set(CIRCLE_RECIPE_REPO "${CircleRecipes_DIR}")
set(TEST_RECIPE_FILENAME "test.recipe")
set(TEST_RULE_FILENAME "test.rule")

set(MODEL2NNPKG "${NNAS_PROJECT_SOURCE_DIR}/tools/nnpackage_tool/model2nnpkg/model2nnpkg.sh")
# Get test case list
unset(RECIPES)
file(GLOB TFLITE_SUBDIR RELATIVE ${TFLITE_RECIPE_REPO} ${TFLITE_RECIPE_REPO}/*)
foreach(DIR IN ITEMS ${TFLITE_SUBDIR})
  if(IS_DIRECTORY ${TFLITE_RECIPE_REPO}/${DIR})
    list(APPEND RECIPES ${DIR})
  endif()
endforeach()
file(GLOB CIRCLE_SUBDIR RELATIVE ${CIRCLE_RECIPE_REPO} ${CIRCLE_RECIPE_REPO}/*)
foreach(DIR IN ITEMS ${CIRCLE_SUBDIR})
  if(IS_DIRECTORY ${CIRCLE_RECIPE_REPO}/${DIR})
    list(APPEND RECIPES ${DIR})
  endif()
endforeach()

macro(circlize NAME)
  set(NO_CIRCLIZE_${NAME} TRUE)
  set(NO_OPTIMIZE_${NAME} TRUE)
  set(NO_TCGEN_${NAME} TRUE)
endmacro()
macro(optimize NAME)
  set(NO_OPTIMIZE_${NAME} TRUE)
endmacro()
macro(tcgenerate NAME)
  set(NO_TCGEN_${NAME} TRUE)
endmacro()

include("exclude.lst")

foreach(RECIPE IN ITEMS ${RECIPES})
  unset(OPT_FORMAT)
  unset(MODEL_FORMAT)

  set(RECIPE_FILE "${RECIPE}.recipe")
  set(RULE_FILE "${RECIPE}.rule")
  set(TFLITE_RECIPE_SOURCE_PATH "${TFLITE_RECIPE_REPO}/${RECIPE}/${TEST_RECIPE_FILENAME}")
  set(CIRCLE_RECIPE_SOURCE_PATH "${CIRCLE_RECIPE_REPO}/${RECIPE}/${TEST_RECIPE_FILENAME}")

  if(NOT EXISTS "${TFLITE_RECIPE_SOURCE_PATH}")
    if(NOT EXISTS "${CIRCLE_RECIPE_SOURCE_PATH}")
      message(FATAL_ERROR "Missing recipe of '${RECIPE}' test")
    else()
      # circle recipe
      set(MODEL_FORMAT "circle")
      set(RECIPE_SOURCE_PATH ${CIRCLE_RECIPE_SOURCE_PATH})
    endif()
  else()
    # tflite recipe
    set(MODEL_FORMAT "tflite")
    set(RECIPE_SOURCE_PATH ${TFLITE_RECIPE_SOURCE_PATH})
  endif()

  set(TFLITE_RULE_SOURCE_PATH "${TFLITE_RECIPE_REPO}/${RECIPE}/${TEST_RULE_FILENAME}")
  set(CIRCLE_RULE_SOURCE_PATH "${CIRCLE_RECIPE_REPO}/${RECIPE}/${TEST_RULE_FILENAME}")

  unset(RULE_SOURCE_PATH)
  if(EXISTS "${TFLITE_RULE_SOURCE_PATH}")
    set(RULE_SOURCE_PATH ${TFLITE_RULE_SOURCE_PATH})
  endif()
  if(EXISTS "${CIRCLE_RULE_SOURCE_PATH}")
    set(RULE_SOURCE_PATH ${CIRCLE_RULE_SOURCE_PATH})
  endif()

  set(RECIPE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}")
  set(RULE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RULE_FILE}")

  set(TFLITE_FILE "${RECIPE}.tflite")
  set(TFLITE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TFLITE_FILE}")
  set(CIRCLE_FILE "${RECIPE}.circle")
  set(CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CIRCLE_FILE}")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_BINARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${RECIPE_SOURCE_PATH}" "${RECIPE_BINARY_PATH}"
    DEPENDS ${RECIPE_SOURCE_PATH}
    COMMENT "Generate ${RECIPE_FILE}"
  )
  list(APPEND TEST_DEPS ${RECIPE_BINARY_PATH})

  if(DEFINED RULE_SOURCE_PATH)
    # Copy .rule
    add_custom_command(OUTPUT ${RULE_BINARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${RULE_SOURCE_PATH}" "${RULE_BINARY_PATH}"
    DEPENDS ${RULE_SOURCE_PATH}
    COMMENT "Generate ${RULE_FILE}"
    )
    list(APPEND TEST_DEPS ${RULE_BINARY_PATH})
  endif()

  if(${MODEL_FORMAT} STREQUAL "tflite")
    # Generate .tflite
    add_custom_command(OUTPUT ${TFLITE_OUTPUT_PATH}
      COMMAND $<TARGET_FILE:tflchef-file> ${RECIPE_BINARY_PATH} ${TFLITE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:tflchef-file> ${RECIPE_BINARY_PATH}
      COMMENT "Generate ${TFLITE_FILE}"
    )
    list(APPEND TEST_DEPS ${TFLITE_OUTPUT_PATH})

    if(NOT DEFINED NO_CIRCLIZE_${RECIPE})
    # Generate .circle
    add_custom_command(OUTPUT ${CIRCLE_OUTPUT_PATH}
      COMMAND $<TARGET_FILE:tflite2circle> ${TFLITE_OUTPUT_PATH} ${CIRCLE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:tflite2circle> ${TFLITE_OUTPUT_PATH}
      COMMENT "Generate ${CIRCLE_FILE}"
    )
    set(MODEL_FORMAT "circle")
    list(APPEND TEST_DEPS ${CIRCLE_OUTPUT_PATH})
    endif()
  else()
    # Generate .circle
    add_custom_command(OUTPUT ${CIRCLE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:circlechef-file> ${RECIPE_BINARY_PATH} ${CIRCLE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:circlechef-file> ${RECIPE_BINARY_PATH}
    COMMENT "Generate ${CIRCLE_FILE}"
    )
    list(APPEND TEST_DEPS ${CIRCLE_OUTPUT_PATH})
  endif()

  set(OPT_CIRCLE_FILE "${RECIPE}.opt.circle")
  set(OPT_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${OPT_CIRCLE_FILE}")

  if(NOT DEFINED NO_OPTIMIZE_${RECIPE})
    # Generate optimized .circle
    add_custom_command(OUTPUT ${OPT_CIRCLE_OUTPUT_PATH}
      COMMAND $<TARGET_FILE:circle2circle> --all ${CIRCLE_OUTPUT_PATH} ${OPT_CIRCLE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:circle2circle>  ${CIRCLE_OUTPUT_PATH}
      COMMENT "Generate ${OPT_CIRCLE_FILE}"
    )
    set(OPT_FORMAT ".opt")
    list(APPEND TEST_DEPS ${OPT_CIRCLE_OUTPUT_PATH})
  endif()

  set(MODEL_FILE "${RECIPE}${OPT_FORMAT}.${MODEL_FORMAT}")
  set(MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_FILE}")
  set(NNPKG_FILE "${RECIPE}")
  set(NNPKG_PATH "${CMAKE_CURRENT_BINARY_DIR}/${NNPKG_FILE}")

  add_custom_command(OUTPUT ${NNPKG_PATH}
    COMMAND ${MODEL2NNPKG} -p ${RECIPE} ${MODEL_PATH}
    DEPENDS ${MODEL2NNPKG} ${MODEL_PATH}
    COMMENT "Generate ${RECIPE} nnpackage"
  )
  list(APPEND TEST_DEPS ${NNPKG_PATH})

  set(INPUT_HDF5_FILE "${RECIPE}${OPT_FORMAT}.input.h5")
  set(INPUT_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${INPUT_HDF5_FILE}")

  set(EXPECTED_HDF5_FILE "${RECIPE}${OPT_FORMAT}.expected.h5")
  set(EXPECTED_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${EXPECTED_HDF5_FILE}")

  if(NOT DEFINED NO_TCGEN_${RECIPE})
    # Generate input.h5, expected.h5
    add_custom_command(OUTPUT ${INPUT_BIN_PATH} ${EXPECTED_BIN_PATH}
      COMMAND $<TARGET_FILE:testDataGenerator> ${MODEL_FILE}
      DEPENDS $<TARGET_FILE:testDataGenerator> ${MODEL_FILE}
      COMMENT "Generate ${INPUT_BIN_PATH} and ${EXPECTED_BIN_PATH}"
    )

    # Generate test directory
    set(TC_DIRECTORY "${NNPKG_PATH}/metadata/tc")
    add_custom_command(OUTPUT ${TC_DIRECTORY}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${TC_DIRECTORY}
      DEPENDS ${NNPKG_PATH}
      COMMENT "Generate ${RECIPE} nnpackage test directory"
    )

    # Move input hdf5 file to test directory
    set(INPUT_NNPKG_PATH "${TC_DIRECTORY}/input.h5")
    add_custom_command(OUTPUT ${INPUT_NNPKG_PATH}
      COMMAND ${CMAKE_COMMAND} -E rename ${INPUT_BIN_PATH} ${INPUT_NNPKG_PATH}
      DEPENDS ${INPUT_BIN_PATH} ${TC_DIRECTORY}
      COMMENT "Move ${INPUT_HDF5_FILE} to nnpackage"
    )

    # Move expected hdf5 file to test directory
    set(EXPECTED_NNPKG_PATH "${TC_DIRECTORY}/expected.h5")
    add_custom_command(OUTPUT ${EXPECTED_NNPKG_PATH}
      COMMAND ${CMAKE_COMMAND} -E rename ${EXPECTED_BIN_PATH} ${EXPECTED_NNPKG_PATH}
      DEPENDS ${EXPECTED_BIN_PATH} ${TC_DIRECTORY}
      COMMENT "Move ${EXPECTED_HDF5_FILE} to nnpackage"
    )
    list(APPEND TEST_DEPS ${TC_DIRECTORY} ${INPUT_BIN_PATH} ${EXPECTED_BIN_PATH}
                          ${INPUT_NNPKG_PATH} ${EXPECTED_NNPKG_PATH})
  endif()
endforeach()

add_custom_target(common_artifacts_deps ALL DEPENDS ${TEST_DEPS})
