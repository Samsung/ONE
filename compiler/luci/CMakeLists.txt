# Some targets do not support dynamic linking: MCU, TrustZone applications, etc.
# STATIC_LUCI option allows us to compile luci and luci related components safely
# and suppress various cmake warnings.
#
# Currently this feature is used for luci-interpreter MCU builds.
if (STATIC_LUCI)
  set(LIBRARY_TYPE "STATIC")
else()
  set(LIBRARY_TYPE "SHARED")
endif()

add_subdirectory(env)
add_subdirectory(log)
add_subdirectory(lang)
add_subdirectory(logex)
add_subdirectory(testhelper)
add_subdirectory(service)
add_subdirectory(pass)
add_subdirectory(profile)
add_subdirectory(partition)
add_subdirectory(import)
add_subdirectory(export)
add_subdirectory(tester)

add_subdirectory(tests)
