set(MICRO_READER_SOURCE
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/reader/CircleMicroReader.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/reader/CircleMicroReaderHelper.h" "CircleMicroReader.cpp"
    "CircleMicroReaderHelper.cpp")

add_library("luci_micro_circle_reader${READER_SUFFIX}" STATIC ${MICRO_READER_SOURCE})
target_link_libraries("luci_micro_circle_reader${READER_SUFFIX}" PUBLIC luci_micro_circle_schema)

target_include_directories("luci_micro_circle_reader${READER_SUFFIX}" PUBLIC "${GENERATED_INCLUDE_DIR}")
target_include_directories("luci_micro_circle_reader${READER_SUFFIX}" PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
