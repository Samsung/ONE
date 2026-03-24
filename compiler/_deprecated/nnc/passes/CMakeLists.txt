#
# COMMON UTILS
#
set(DEF_CONV ${NNC_ROOT_SRC_DIR}/utils/def2src.cpp)
add_executable(def2src ${DEF_CONV})
add_subdirectory(dot_dumper)

#
# MIDDLE PASSES
#
add_subdirectory(optimizations)
add_subdirectory(transformations)
