unset(OUTPUT_FILES)
set(OUTPUT_FILES "${ONERT_MICRO_EXTERNAL_DIR}/gen/circle-generated/circle/schema_generated.h")
set(TGT luci_micro_circle_schema)

# NOTE This header-only library
add_library(${TGT} STATIC ${OUTPUT_FILES})
set_target_properties(${TGT} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${TGT} PUBLIC "${ONERT_MICRO_EXTERNAL_DIR}/gen")
target_include_directories(${TGT} PUBLIC "${ONERT_MICRO_EXTERNAL_DIR}")
