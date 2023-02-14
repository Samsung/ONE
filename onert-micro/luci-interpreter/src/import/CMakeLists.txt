set(SOURCES "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/GraphBuilderRegistry.h" GraphBuilderRegistry.cpp)

# include specific builders
file(GLOB_RECURSE NODES "Nodes/*")
list(APPEND SOURCES ${NODES})

add_library(${LUCI_INTERPRETER_IMPORT} STATIC ${SOURCES})
if(NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(${LUCI_INTERPRETER_IMPORT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)

target_include_directories(${LUCI_INTERPRETER_IMPORT} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_IMPORT} PUBLIC luci_import)
