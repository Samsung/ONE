# python3 venv folder
# NOTE Docker image for CI doesn't use venv
set(VENV_PATH "${CMAKE_SOURCE_DIR}/infra/overlay/venv")

# Copy test scripts
unset(GEN_SCRIPT_DEPS )
macro(COPY_SCRIPT FILENAME)
  set(SCRIPT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
  set(SCRIPT_DST "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
  add_custom_command(
    OUTPUT ${SCRIPT_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${SCRIPT_SRC}" "${SCRIPT_DST}"
    DEPENDS ${SCRIPT_SRC}
    COMMENT "gen-onnx: prepare ${FILENAME}"
  )
  list(APPEND GEN_SCRIPT_DEPS "${SCRIPT_DST}")
endmacro(COPY_SCRIPT)

COPY_SCRIPT(run_gen_onnx.sh)
COPY_SCRIPT(run_gen_onnx.py)

# Models folder for unit testing
set(PYTORCH_UNIT_PATH "${CMAKE_SOURCE_DIR}/models/unit")
set(PYTORCH_NET_PATH "${CMAKE_SOURCE_DIR}/models/net")

# Pytorch script files
unset(PYTORCH_UNIT_ITEMS )
# gather all the names in /models/unit
file(GLOB PYTORCH_MODELS_ITEMS RELATIVE ${PYTORCH_UNIT_PATH} ${PYTORCH_UNIT_PATH}/*)
# for each name, if the item is a folder, add to the list
foreach(PYTORCH_ITEM IN ITEMS ${PYTORCH_MODELS_ITEMS})
  if(IS_DIRECTORY ${PYTORCH_UNIT_PATH}/${PYTORCH_ITEM})
    if(EXISTS ${PYTORCH_UNIT_PATH}/${PYTORCH_ITEM}/__init__.py)
      list(APPEND PYTORCH_UNIT_ITEMS ${PYTORCH_ITEM})
    endif()
  endif()
endforeach()

unset(PYTORCH_MODELS_ITEMS)
unset(PYTORCH_NET_ITEMS )
# gather all the names in /models/net
file(GLOB PYTORCH_MODELS_ITEMS RELATIVE ${PYTORCH_NET_PATH} ${PYTORCH_NET_PATH}/*)
# for each name, if the item is a folder, add to the list
foreach(PYTORCH_ITEM IN ITEMS ${PYTORCH_MODELS_ITEMS})
  if(IS_DIRECTORY ${PYTORCH_NET_PATH}/${PYTORCH_ITEM})
    if(EXISTS ${PYTORCH_NET_PATH}/${PYTORCH_ITEM}/__init__.py)
      list(APPEND PYTORCH_NET_ITEMS ${PYTORCH_ITEM})
    endif()
  endif()
endforeach()

unset(GEN_FILES_DEPS)
# for each items in the list, run run_gen_onnx to generate onnx model
foreach(PYTORCH_ITEM IN ITEMS ${PYTORCH_UNIT_ITEMS})
  set(ONNX_FILE ${PYTORCH_ITEM}.onnx)
  add_custom_command(OUTPUT ${ONNX_FILE}
    COMMAND bash run_gen_onnx.sh
            ${VENV_PATH} ${PYTORCH_UNIT_PATH} ${PYTORCH_ITEM} ${ONNX_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/run_gen_onnx.sh
            ${CMAKE_CURRENT_BINARY_DIR}/run_gen_onnx.py
            ${PYTORCH_UNIT_PATH}/${PYTORCH_ITEM}/__init__.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generate ${ONNX_FILE}"
  )
  list(APPEND GEN_FILES_DEPS ${ONNX_FILE})
endforeach()

foreach(PYTORCH_ITEM IN ITEMS ${PYTORCH_NET_ITEMS})
  set(ONNX_FILE ${PYTORCH_ITEM}.onnx)
  add_custom_command(OUTPUT ${ONNX_FILE}
    COMMAND bash run_gen_onnx.sh
            ${VENV_PATH} ${PYTORCH_NET_PATH} ${PYTORCH_ITEM} ${ONNX_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/run_gen_onnx.sh
            ${CMAKE_CURRENT_BINARY_DIR}/run_gen_onnx.py
            ${PYTORCH_NET_PATH}/${PYTORCH_ITEM}/__init__.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generate ${ONNX_FILE}"
  )
  list(APPEND GEN_FILES_DEPS ${ONNX_FILE})
endforeach()

# gen_onnx_target is used to make build dependency
add_custom_target(gen_onnx_target ALL DEPENDS ${GEN_SCRIPT_DEPS} ${GEN_FILES_DEPS})

set(GEN_ONNX_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
