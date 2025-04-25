function(_VulkanSource_import)
  if(NOT ${DOWNLOAD_VULKAN})
    set(VulkanSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_VULKAN})

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(VULKAN_URL ${EXTERNAL_DOWNLOAD_SERVER}/KhronosGroup/Vulkan-Headers/archive/ec2db85225ab410bc6829251bef6c578aaed5868.tar.gz)
  ExternalSource_Download(VULKAN
    DIRNAME VULKAN
    URL ${VULKAN_URL})

  set(VulkanSource_DIR ${VULKAN_SOURCE_DIR} PARENT_SCOPE)
  set(VulkanSource_FOUND TRUE PARENT_SCOPE)
endfunction(_VulkanSource_import)

_VulkanSource_import()
