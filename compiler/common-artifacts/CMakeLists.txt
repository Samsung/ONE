#[[ Generate common python virtual enviornment ]]
# NOTE find_package try to use at least python3.10 as follows depending on platform version
#   Ubuntu20.04; explictly installed python3.10 (default is python3.8)
#   Ubuntu22.04; default python3.10
#   Ubuntu24.04; default python3.12
#   refer https://github.com/Samsung/ONE/issues/9962
#   refer https://github.com/Samsung/ONE/issues/15226
# find python 3.10 or above
find_package(Python 3.10 EXACT COMPONENTS Interpreter QUIET)
if(NOT Python_FOUND)
  find_package(Python 3.10 COMPONENTS Interpreter QUIET)
endif()

if(NOT Python_Interpreter_FOUND)
  message(STATUS "Build common-artifacts: FAILED (Python3 is missing)")
  return()
endif()

# NOTE assume only use 3.10.x or 3.12.x
if((Python_VERSION VERSION_GREATER_EQUAL 3.10) AND (Python_VERSION VERSION_LESS 3.11))
  set(PYTHON_VERSION_MINOR 10)
elseif((Python_VERSION VERSION_GREATER_EQUAL 3.12) AND (Python_VERSION VERSION_LESS 3.13))
  set(PYTHON_VERSION_MINOR 12)
else()
  # TODO support more
  message(STATUS "Build common-artifacts: FAILED (Unsupported python: ${Python_VERSION})")
  return()
endif()

# NOTE ONE_PYTHON_VERSION_MINOR is to provide current python3.x minor version
# number to modules that use common artifacts.
set(ONE_PYTHON_VERSION_MINOR ${PYTHON_VERSION_MINOR} PARENT_SCOPE)

set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

# Create python virtual environment
set(VIRTUALENV_OVERLAY "${NNCC_OVERLAY_DIR}/venv")

add_custom_command(
  OUTPUT ${VIRTUALENV_OVERLAY}
  COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV_OVERLAY}
)

# Create requirements.txt and install required pip packages
set(REQUIREMENTS_FILE "requirements.txt")
set(REQUIREMENTS_OVERLAY_PATH "${VIRTUALENV_OVERLAY}/${REQUIREMENTS_FILE}")

set(PYTHON_OVERLAY python3)
if(PYTHON_EXECUTABLE MATCHES python3.10)
  set(PYTHON_OVERLAY python3.10)
endif()

# NOTE when using behind proxy with self signed certificate, need to set '--trusted-host' options
set(PIP_OPTION_TRUSTED_HOST )
if(DEFINED ENV{ONE_PIP_OPTION_TRUST_HOST})
  set(PIP_OPTION_TRUSTED_HOST --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org)
endif()

# NOTE `tensorflow-cpu` package is not available for aarch64, so we use `tensorflow` package.
# NOTE refer https://github.com/Samsung/ONE/issues/15226 for versions
set(PY_PKG_LIST )
if (PYTHON_VERSION_MINOR EQUAL 8)
  # python3.8; Ubuntu 18.04, 20.04
  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND PY_PKG_LIST "tensorflow==2.12.1")
  else(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND PY_PKG_LIST "tensorflow-cpu==2.12.1")
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  list(APPEND PY_PKG_LIST "flatbuffers==23.5.26")
  list(APPEND PY_PKG_LIST "protobuf==4.23.3")
  list(APPEND PY_PKG_LIST "pydot==1.4.2")
  list(APPEND PY_PKG_LIST "pytest==7.4.3")
  list(APPEND PY_PKG_LIST "h5py==3.11.0")
  list(APPEND PY_PKG_LIST "cffi==1.16.0")
  list(APPEND PY_PKG_LIST "numpy==1.24.3")
elseif (PYTHON_VERSION_MINOR EQUAL 10)
  # python 3.10; Ubuntu 22.04
  # TF2.19.0 release at 2025-03-12
  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND PY_PKG_LIST "tensorflow==2.19.0")
  else(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND PY_PKG_LIST "tensorflow-cpu==2.19.0")
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  list(APPEND PY_PKG_LIST "flatbuffers==24.3.25")
  list(APPEND PY_PKG_LIST "protobuf==4.23.3")
  list(APPEND PY_PKG_LIST "pydot==1.4.2")
  list(APPEND PY_PKG_LIST "pytest==7.4.3")
  list(APPEND PY_PKG_LIST "h5py==3.11.0")
  list(APPEND PY_PKG_LIST "cffi==1.16.0")
  list(APPEND PY_PKG_LIST "numpy==1.26.4")
elseif (PYTHON_VERSION_MINOR EQUAL 12)
  # python 3.12; Ubuntu 24.04
  # TF2.19.0 release at 2025-03-12
  if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND PY_PKG_LIST "tensorflow==2.19.0")
  else(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND PY_PKG_LIST "tensorflow-cpu==2.19.0")
  endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  list(APPEND PY_PKG_LIST "flatbuffers==24.3.25")
  list(APPEND PY_PKG_LIST "protobuf==4.23.3")
  list(APPEND PY_PKG_LIST "pydot==1.4.2")
  list(APPEND PY_PKG_LIST "pytest==7.4.3")
  list(APPEND PY_PKG_LIST "h5py==3.11.0")
  list(APPEND PY_PKG_LIST "cffi==1.16.0")
  list(APPEND PY_PKG_LIST "numpy==1.26.4")
else()
  # should not enter here
  message(FATAL_ERROR "Build common-artifacts: FAILED (Invalid pyhon version)")
endif()

add_custom_command(
  OUTPUT REQUIREMENTS_OVERLAY_PATH_clean
  COMMAND ${CMAKE_COMMAND} -E remove -f ${REQUIREMENTS_OVERLAY_PATH}
  DEPENDS ${VIRTUALENV_OVERLAY}
)

set(PY_PKG_LIST_DEPENDS_PREV REQUIREMENTS_OVERLAY_PATH_clean)
foreach(PKG_VER IN LISTS PY_PKG_LIST)
  add_custom_command(
    OUTPUT ${PKG_VER}_py_item ${REQUIREMENTS_OVERLAY_PATH}
    COMMAND ${CMAKE_COMMAND} -E echo "${PKG_VER}" >> ${REQUIREMENTS_OVERLAY_PATH}
    DEPENDS ${VIRTUALENV_OVERLAY_TF} ${PY_PKG_LIST_DEPENDS_PREV}
  )
  set(PY_PKG_LIST_DEPENDS_PREV ${PKG_VER}_py_item)
endforeach()

add_custom_command(
  OUTPUT VIRTUALENV_OVERLAY_installed
  COMMAND ${VIRTUALENV_OVERLAY}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
          ${PIP_OPTION_TRUSTED_HOST} install --upgrade pip setuptools
  COMMAND ${VIRTUALENV_OVERLAY}/bin/${PYTHON_OVERLAY} -m pip --default-timeout=1000
          ${PIP_OPTION_TRUSTED_HOST} install -r ${REQUIREMENTS_OVERLAY_PATH} --upgrade
  DEPENDS ${VIRTUALENV_OVERLAY} ${PY_PKG_LIST_DEPENDS_PREV}
)

add_custom_target(common_artifacts_python_deps ALL
  DEPENDS ${VIRTUALENV_OVERLAY}
          ${REQUIREMENTS_OVERLAY_PATH}
          VIRTUALENV_OVERLAY_installed
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
