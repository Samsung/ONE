set(SUPPORT_SOURCES CommandLine.cpp CLOptionChecker.cpp)

nnc_add_library(nnc_support STATIC ${SUPPORT_SOURCES})
set_target_properties(nnc_support PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(nnc_support PROPERTIES POSITION_INDEPENDENT_CODE ON)
