function(_ArmNoneEabiToolchain_import)
    set(ARM_NONE_EABI_GCC_FILENAME "gcc-arm-none-eabi-9-2020-q2-update-x86_64-linux.tar.bz2")
    set(ARM_NONE_EABI_GCC_TOOLCHAIN_PATH "${NNAS_EXTERNALS_DIR}/gcc-arm-none-eabi-9-2020-q2-update")
    set(ARM_NONE_EABI_URL "https://developer.arm.com/-/media/Files/downloads/gnu-rm/9-2020q2/${ARM_NONE_EABI_GCC_FILENAME}?revision=05382cca-1721-44e1-ae19-1e7c3dc96118&hash=CEB1348BF26C0285FD788E2424773FA304921735")

    if (NOT EXISTS "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}")
        file(DOWNLOAD ${ARM_NONE_EABI_URL} "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2"
                EXPECTED_MD5 2b9eeccc33470f9d3cda26983b9d2dc6 STATUS result)
        if (NOT ${STATUS} EQUAL 0)
            set(ArmNoneEabiToolchain_FOUND FALSE PARENT_SCOPE)
            return()
        endif ()
        execute_process(COMMAND tar -xjf "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2" -C "${NNAS_EXTERNALS_DIR}")
        file(REMOVE "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2")
    endif ()

    set(ArmNoneEabiToolchain_FOUND TRUE PARENT_SCOPE)
    set(ArmNoneEabiToolchain_DIR "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}" PARENT_SCOPE)
    set(ArmNoneEabiToolchain_BINARY_DIR "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin" PARENT_SCOPE)
endfunction()

_ArmNoneEabiToolchain_import()
