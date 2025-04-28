add_library(farmhash "${FarmhashSource_DIR}/src/farmhash.cc")
set_target_properties(farmhash PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(farmhash PUBLIC "${FarmhashSource_DIR}/src")
