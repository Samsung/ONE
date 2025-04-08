list(APPEND CIRCLE_RESIZER_CORE_SOURCES Shape.cpp)
list(APPEND CIRCLE_RESIZER_CORE_SOURCES ShapeParser.cpp)

add_library(circle_resizer_core STATIC "${CIRCLE_RESIZER_CORE_SOURCES}")

target_include_directories(circle_resizer_core PUBLIC ../include)

install(TARGETS circle_resizer_core DESTINATION lib)
