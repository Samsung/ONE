add_library(circle_schema INTERFACE)

nnfw_find_package(FlatBuffers 23.5.26 REQUIRED)

set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS --gen-onefile --gen-object-api)
build_flatbuffers(
  "circle_schema.fbs"   # flatbuffers_schemas
  ""                    # schema_include_dirs
  circle_schema_header  # custom_target_name
  ""                    # additional_dependencies
  "generated"           # generated_includes_dir
  ""                    # binary_schemas_dir
  ""                    # copy_text_schemas_dir
)

set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS --gen-onefile)
build_flatbuffers(
  "circle_traininfo.fbs"  # flatbuffers_schemas
  ""                      # schema_include_dirs
  circle_traininfo_header # custom_target_name
  ""                      # additional_dependencies
  "generated"             # generated_includes_dir
  ""                      # binary_schemas_dir
  ""                      # copy_text_schemas_dir
)

target_link_libraries(circle_schema INTERFACE flatbuffers::flatbuffers)
target_include_directories(circle_schema INTERFACE generated)
add_dependencies(circle_schema circle_schema_header circle_traininfo_header)
