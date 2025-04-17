list(APPEND CIRCLE_RESIZER_SOURCES Dim.cpp)
list(APPEND CIRCLE_RESIZER_SOURCES Shape.cpp)
list(APPEND CIRCLE_RESIZER_SOURCES ShapeParser.cpp)
list(APPEND CIRCLE_RESIZER_SOURCES CircleModel.cpp)
list(APPEND CIRCLE_RESIZER_SOURCES ModelEditor.cpp)

add_library(circle_resizer_core SHARED "${CIRCLE_RESIZER_SOURCES}")

target_include_directories(circle_resizer_core PUBLIC ../include)

target_link_libraries(circle_resizer_core PRIVATE mio_circle08)
target_link_libraries(circle_resizer_core PRIVATE logo_core)
target_link_libraries(circle_resizer_core PRIVATE luci_pass)
target_link_libraries(circle_resizer_core PRIVATE luci_lang)
target_link_libraries(circle_resizer_core PRIVATE luci_export)
target_link_libraries(circle_resizer_core PRIVATE luci_import)
target_link_libraries(circle_resizer_core PRIVATE logo)
target_link_libraries(circle_resizer_core PRIVATE luci_log)

install(TARGETS circle_resizer_core DESTINATION lib)
