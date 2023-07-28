#[[ Generate common python virtual enviornment ]]
# NOTE find_package try to use at least python3.8 as follows depending on platform version
#   Ubuntu18.04; explictly installed python3.8 (default is python3.6)
#   Ubuntu20.04; default python3.8
#   Ubuntu22.04; default python3.10
#   refer https://github.com/Samsung/ONE/issues/9962
find_package(PythonInterp 3.8 QUIET)
find_package(PythonLibs 3.8 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build common-artifacts: FALSE (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 8)
  message(STATUS "Build common-artifacts: FALSE (You need to install Python version higher than 3.8)")
  return()
endif()

# Create python virtual environment with tensorflow 2.8.0
set(VIRTUALENV_OVERLAY_TF_2_8_0 "${NNCC_OVERLAY_DIR}/venv_2_8_0")
# TensorFlow 2.10.1 for Ubuntu22.04
if(ONE_UBUNTU_CODENAME_JAMMY)
  set(VIRTUALENV_OVERLAY_TF_2_10_1 "${NNCC_OVERLAY_DIR}/venv_2_10_1")
endif(ONE_UBUNTU_CODENAME_JAMMY)

add_custom_command(
  OUTPUT ${VIRTUALENV_OVERLAY_TF_2_8_0}
  COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV_OVERLAY_TF_2_8_0}
)

# Create requirements.txt and install required pip packages
set(REQUIREMENTS_FILE "requirements.txt")
set(REQUIREMENTS_OVERLAY_PATH_TF_2_8_0 "${VIRTUALENV_OVERLAY_TF_2_8_0}/${REQUIREMENTS_FILE}")

if(ONE_UBUNTU_CODENAME_JAMMY)
  add_custom_command(
    OUTPUT ${VIRTUALENV_OVERLAY_TF_2_10_1}
    COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV_OVERLAY_TF_2_10_1}
  )
  set(REQUIREMENTS_FILE "requirements.txt")
  set(REQUIREMENTS_OVERLAY_PATH_TF_2_10_1 "${VIRTUALENV_OVERLAY_TF_2_10_1}/${REQUIREMENTS_FILE}")
endif(ONE_UBUNTU_CODENAME_JAMMY)

set(PYTHON_OVERLAY python3)
if(PYTHON_EXECUTABLE MATCHES python3.8)
  set(PYTHON_OVERLAY python3.8)
endif()

# NOTE when using behind proxy with self signed certificate, need to set '--trusted-host' options
set(PIP_OPTION_TRUSTED_HOST )
if(DEFINED ENV{ONE_PIP_OPTION_TRUST_HOST})
  set(PIP_OPTION_TRUSTED_HOST --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org)
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  # NOTE installing tensorflow 2.10.1 in OVERLAY_PATH_TF_2_8_0 is strange, but keeping different
  # versions of tensorflow is temporary and it is a way to support aarch64 native build while
  # minimizing the impact on the entire build system, so we choose this way.
  # NOTE `tensorflow-cpu` package is not available for aarch64, so we use `tensorflow` package.
  # TODO renaming OVERLAY_PATH_TF_2_8_0 and OVERLAY_PATH_TF_2_10_1 to version neutral names would
  # be appropriate.
  add_custom_command(
    OUTPUT ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "tensorflow==2.10.1" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "flatbuffers==23.1.21" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "protobuf==3.19.6" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "pydot==1.4.2" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${VIRTUALENV_OVERLAY_TF_2_8_0}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
            ${PIP_OPTION_TRUSTED_HOST} install --upgrade pip setuptools
    COMMAND ${VIRTUALENV_OVERLAY_TF_2_8_0}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
            ${PIP_OPTION_TRUSTED_HOST} install -r ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0} --upgrade
    DEPENDS ${VIRTUALENV_OVERLAY_TF_2_8_0}
  )
else(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  # NOTE refer https://github.com/protocolbuffers/protobuf/issues/10051
  # TODO remove protobuf==3.20.1 when issue is resolved
  add_custom_command(
    OUTPUT ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "tensorflow-cpu==2.8.0" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "flatbuffers==1.12" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "protobuf==3.20.1" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${CMAKE_COMMAND} -E echo "pydot==1.4.2" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
    COMMAND ${VIRTUALENV_OVERLAY_TF_2_8_0}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
            ${PIP_OPTION_TRUSTED_HOST} install --upgrade pip setuptools
    COMMAND ${VIRTUALENV_OVERLAY_TF_2_8_0}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
            ${PIP_OPTION_TRUSTED_HOST} install -r ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0} --upgrade
    DEPENDS ${VIRTUALENV_OVERLAY_TF_2_8_0}
  )
endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")

add_custom_target(common_artifacts_python_deps ALL
  DEPENDS ${VIRTUALENV_OVERLAY_TF_2_8_0}
          ${REQUIREMENTS_OVERLAY_PATH_TF_2_8_0}
)

if(ONE_UBUNTU_CODENAME_JAMMY)
  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  # tensorflow-cpu is not available on aarch64
    set(TENSORFLOW_PKG_VER "tensorflow==2.10.1")
  else(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(TENSORFLOW_PKG_VER "tensorflow-cpu==2.10.1")
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  add_custom_command(
    OUTPUT ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
    COMMAND ${CMAKE_COMMAND} -E echo ${TENSORFLOW_PKG_VER} >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
    COMMAND ${CMAKE_COMMAND} -E echo "flatbuffers==23.1.21" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
    COMMAND ${CMAKE_COMMAND} -E echo "protobuf==3.19.6" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
    COMMAND ${CMAKE_COMMAND} -E echo "pydot==1.4.2" >> ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
    COMMAND ${VIRTUALENV_OVERLAY_TF_2_10_1}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
            ${PIP_OPTION_TRUSTED_HOST} install --upgrade pip setuptools
    COMMAND ${VIRTUALENV_OVERLAY_TF_2_10_1}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
            ${PIP_OPTION_TRUSTED_HOST} install -r ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1} --upgrade
    DEPENDS ${VIRTUALENV_OVERLAY_TF_2_10_1}
  )

  add_custom_target(common_artifacts_python_u22_deps ALL
    DEPENDS ${VIRTUALENV_OVERLAY_TF_2_10_1}
            ${REQUIREMENTS_OVERLAY_PATH_TF_2_10_1}
  )
endif(ONE_UBUNTU_CODENAME_JAMMY)

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
target_link_libraries(testDataGenerator PRIVATE safemain)

unset(TEST_DEPS)

# Include recipe repo
nncc_find_resource(TensorFlowLiteRecipes)
nncc_find_resource(CircleRecipes)
set(TFLITE_RECIPE_REPO "${TensorFlowLiteRecipes_DIR}")
set(CIRCLE_RECIPE_REPO "${CircleRecipes_DIR}")
set(TEST_RECIPE_FILENAME "test.recipe")
set(TEST_RULE_FILENAME "test.rule")
set(TEST_QCONFIG_FILENAME "test.qconf.json")

set(MODEL2NNPKG "${NNAS_PROJECT_SOURCE_DIR}/tools/nnpackage_tool/model2nnpkg/model2nnpkg.py")
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

# TODO revise using variadic arguments
macro(tcgenerate_option NAME OPTION ARG1 ARG2 ARG3)
  set(TCGEN_OPT_${NAME} ${OPTION} ${ARG1} ${ARG2} ${ARG3})
endmacro()

include("options.lst")

foreach(RECIPE IN ITEMS ${RECIPES})
  unset(OPT_FORMAT)
  unset(MODEL_FORMAT)

  set(RECIPE_FILE "${RECIPE}.recipe")
  set(RULE_FILE "${RECIPE}.rule")
  set(QCONFIG_FILE "${RECIPE}.qconf.json")
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

  set(TFLITE_QCONFIG_SOURCE_PATH "${TFLITE_RECIPE_REPO}/${RECIPE}/${TEST_QCONFIG_FILENAME}")
  set(CIRCLE_QCONFIG_SOURCE_PATH "${CIRCLE_RECIPE_REPO}/${RECIPE}/${TEST_QCONFIG_FILENAME}")

  unset(QCONFIG_SOURCE_PATH)
  if(EXISTS "${TFLITE_QCONFIG_SOURCE_PATH}")
    set(QCONFIG_SOURCE_PATH ${TFLITE_QCONFIG_SOURCE_PATH})
  endif()
  if(EXISTS "${CIRCLE_QCONFIG_SOURCE_PATH}")
    set(QCONFIG_SOURCE_PATH ${CIRCLE_QCONFIG_SOURCE_PATH})
  endif()

  set(RECIPE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}")
  set(RULE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RULE_FILE}")
  set(QCONFIG_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${QCONFIG_FILE}")

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

  if(DEFINED QCONFIG_SOURCE_PATH)
    # Copy .qconf.json
    add_custom_command(OUTPUT ${QCONFIG_BINARY_PATH}
      COMMAND ${CMAKE_COMMAND} -E copy "${QCONFIG_SOURCE_PATH}" "${QCONFIG_BINARY_PATH}"
      DEPENDS ${QCONFIG_SOURCE_PATH}
      COMMENT "Generate ${QCONFIG_FILE}"
    )
    list(APPEND TEST_DEPS ${QCONFIG_BINARY_PATH})
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
      # NOTE --resolve_customop_add is just to added for old -O1, no particular meaning
      #      --fold_dequantize is added to fold Tensor(FLOAT16) + DEQUANTIZE (Net_Dequantize_Add)
      #      model. FLOAT16 in general is NOT supported but only Tensor(FLOAT16) + DEQUANTIZE
      #      sequence accepted as folded to Tensor(FLOAT32).
      # TODO revise giving options from the list file
      COMMAND $<TARGET_FILE:circle2circle> --resolve_customop_add --fold_dequantize --fold_densify
              ${CIRCLE_OUTPUT_PATH} ${OPT_CIRCLE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:circle2circle>  ${CIRCLE_OUTPUT_PATH}
      COMMENT "Generate ${OPT_CIRCLE_FILE}"
    )
    set(OPT_FORMAT ".opt")
    list(APPEND TEST_DEPS ${OPT_CIRCLE_OUTPUT_PATH})
  endif()

  set(MODEL_FILE "${RECIPE}${OPT_FORMAT}.${MODEL_FORMAT}")
  set(MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_FILE}")
  set(NNPKG_FILE "${RECIPE}${OPT_FORMAT}")
  set(NNPKG_DIR "${CMAKE_CURRENT_BINARY_DIR}/${NNPKG_FILE}")
  set(NNPKG_MODEL "${NNPKG_DIR}/${MODEL_FILE}")

  # Generate nnpackage directory
  add_custom_command(OUTPUT ${NNPKG_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${NNPKG_DIR}
      DEPENDS ${MODEL_PATH}
      COMMENT "Generate ${RECIPE} nnpackage directory"
    )
  list(APPEND TEST_DEPS ${NNPKG_DIR})

  add_custom_command(OUTPUT ${NNPKG_MODEL}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL2NNPKG} -m ${MODEL_PATH}
    DEPENDS ${MODEL2NNPKG} ${MODEL_PATH}
    COMMENT "Generate ${RECIPE} nnpackage"
  )
  list(APPEND TEST_DEPS ${NNPKG_MODEL})

  if(NOT DEFINED NO_TCGEN_${RECIPE})
    # Generate test directory
    set(TC_DIRECTORY "${NNPKG_DIR}/metadata/tc")
    add_custom_command(OUTPUT ${TC_DIRECTORY}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${TC_DIRECTORY}
      COMMENT "Generate ${RECIPE} nnpackage test directory"
    )
    list(APPEND TEST_DEPS ${TC_DIRECTORY})

    # set ADDITIONAL_OPTIONS as empty (one space before closing is intentional)
    set(ADDITIONAL_OPTIONS )
    if(DEFINED TCGEN_OPT_${RECIPE})
      set(ADDITIONAL_OPTIONS ${ADDITIONAL_OPTIONS} ${TCGEN_OPT_${RECIPE}})
    endif()

    # Generate input.h5, expected.h5
    set(INPUT_HDF5_FILE "${TC_DIRECTORY}/input.h5")
    set(EXPECTED_HDF5_FILE "${TC_DIRECTORY}/expected.h5")
    add_custom_command(OUTPUT ${INPUT_HDF5_FILE} ${EXPECTED_HDF5_FILE}
      COMMAND $<TARGET_FILE:testDataGenerator>
              --input_data ${INPUT_HDF5_FILE}
              --expected_data ${EXPECTED_HDF5_FILE}
              ${ADDITIONAL_OPTIONS}
              ${MODEL_FILE}
      DEPENDS $<TARGET_FILE:testDataGenerator> ${MODEL_FILE} ${TC_DIRECTORY}
      COMMENT "Generate input.h5 and expected.h5 in ${NNPKG_FILE}/metadata/tc"
    )
    list(APPEND TEST_DEPS ${INPUT_HDF5_FILE} ${EXPECTED_HDF5_FILE})
  endif()
endforeach()

add_custom_target(common_artifacts_deps ALL DEPENDS ${TEST_DEPS})
