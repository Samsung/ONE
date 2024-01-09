unset(SOURCES)
set(SOURCES "${ONERT_MICRO_EXTERNAL_DIR}/gen/circle-generated/circle/schema_generated.h")

# NOTE This header-only library
add_library(${OM_CIRCLE_SCHEMA} STATIC ${OUTPUT_FILES})
set_target_properties(${OM_CIRCLE_SCHEMA} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${OM_CIRCLE_SCHEMA} PUBLIC "${ONERT_MICRO_EXTERNAL_DIR}/gen")
target_include_directories(${OM_CIRCLE_SCHEMA} PUBLIC "${ONERT_MICRO_EXTERNAL_DIR}")
