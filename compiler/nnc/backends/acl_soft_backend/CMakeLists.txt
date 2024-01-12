set(ACL_SOFT_BACKEND_CPP_SOURCES AclCppGenerator.cpp AclCppOpGenerator.cpp
    ArtifactGeneratorCppCode.cpp ArtifactGeneratorCppDecl.cpp ArtifactModel.cpp)

file(GLOB_RECURSE ACL_IN_SOURCES "*.in")
nnc_make_generated_sources("${ACL_IN_SOURCES}" ${CMAKE_CURRENT_BINARY_DIR} ACL_GENERATED_SOURCES)

nnc_add_library(acl_soft_backend_cpp SHARED ${ACL_SOFT_BACKEND_CPP_SOURCES} ${ACL_GENERATED_SOURCES})
target_include_directories(acl_soft_backend_cpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(acl_soft_backend_cpp PRIVATE mir)
if(ONE_UBUNTU_CODENAME_BIONIC)
    target_link_libraries(acl_soft_backend_cpp PRIVATE stdc++fs)
endif()

# install soft backend c++ library
nnc_install_library(acl_soft_backend_cpp)
