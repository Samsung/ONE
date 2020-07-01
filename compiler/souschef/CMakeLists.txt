file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(souschef STATIC ${SOURCES})
set_target_properties(souschef PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(souschef PUBLIC include)
