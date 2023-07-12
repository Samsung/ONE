# Assume that luci and related libraries and headers are installed on overlay directory

set(Luci_FOUND FALSE)

find_path(LUCI_HEADERS
    NAMES loco.h luci/IR/CircleNode.h
    PATHS ${EXT_OVERLAY_DIR}/include)

macro(_load_library LUCI_NAME)
    add_library(luci::${LUCI_NAME} SHARED IMPORTED)
    find_library(LUCI_LIB_PATH_${LUCI_NAME} NAMES luci_${LUCI_NAME} PATHS ${EXT_OVERLAY_DIR}/lib)
    if (NOT LUCI_LIB_PATH_${LUCI_NAME})
        return()
    endif()
    set_target_properties(luci::${LUCI_NAME} PROPERTIES
        IMPORTED_LOCATION ${LUCI_LIB_PATH_${LUCI_NAME}}
        INTERFACE_INCLUDE_DIRECTORIES ${LUCI_HEADERS})
endmacro()

_load_library(env)
_load_library(export)
_load_library(import)
_load_library(lang)
_load_library(logex)
_load_library(log)
_load_library(partition)
_load_library(pass)
_load_library(plan)
_load_library(profile)
_load_library(service)

# Need luci::loco to avoid "DSO missing from command line" link error
# TODO Find better way to do this
add_library(luci::loco SHARED IMPORTED)
find_library(LOCO_LIB_PATH NAMES loco PATHS ${EXT_OVERLAY_DIR}/lib)
if (NOT LOCO_LIB_PATH)
    return()
endif()
set_target_properties(luci::loco PROPERTIES
    IMPORTED_LOCATION ${LOCO_LIB_PATH}
    INTERFACE_INCLUDE_DIRECTORIES ${LUCI_HEADERS})

set(Luci_FOUND TRUE)
