nnas_find_package(Boost REQUIRED COMPONENTS filesystem)

set(SOFT_BACKEND_CPP_SOURCES CPPGenerator.cpp ModelAnalyzer.cpp SBSerializer.cpp SequencedIR.cpp)

file(GLOB_RECURSE SOFT_DEF_SOURCES "*.def")

nnc_make_generated_sources("${SOFT_DEF_SOURCES}" ${CMAKE_CURRENT_BINARY_DIR} SOFT_GENERATED_SOURCES)

nnc_add_library(soft_backend_cpp SHARED ${SOFT_BACKEND_CPP_SOURCES} ${SOFT_GENERATED_SOURCES})
target_include_directories(soft_backend_cpp PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(soft_backend_cpp PRIVATE mir ${Boost_LIBRARIES})

# install soft backend c++ library
nnc_install_library(soft_backend_cpp)
