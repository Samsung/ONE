set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-eval-diff ${DRIVER} ${SOURCES})
target_include_directories(circle-eval-diff PRIVATE include)

target_link_libraries(circle-eval-diff arser)
target_link_libraries(circle-eval-diff safemain)
target_link_libraries(circle-eval-diff vconone)
