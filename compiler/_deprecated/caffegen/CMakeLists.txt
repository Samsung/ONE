nnas_find_package(Caffe QUIET)

if(NOT Caffe_FOUND)
  return()
endif(NOT Caffe_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(caffegen ${SOURCES})
target_link_libraries(caffegen cli)
# NOTE "Caffe" package provides both caffe and caffeproto target
# NOTE "caffeproto" is linked to "caffe"
target_link_libraries(caffegen caffe)
