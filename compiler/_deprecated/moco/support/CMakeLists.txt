file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(moco_support SHARED ${SOURCES})
target_include_directories(moco_support PRIVATE src)
target_include_directories(moco_support PUBLIC include)
target_link_libraries(moco_support PUBLIC loco)
target_link_libraries(moco_support PUBLIC moco_lang)
target_link_libraries(moco_support PRIVATE oops)
install(TARGETS moco_support DESTINATION lib)
