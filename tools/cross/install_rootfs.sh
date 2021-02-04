#!/usr/bin/env bash
set -x

usage()
{
    echo "Usage: $0 [BuildArch] [LinuxCodeName] [--setproxy=IP] [--skipunmount]"
    echo "BuildArch can be: arm(default), aarch64 and armel"
    echo "LinuxCodeName - optional, Code name for Linux, can be: xenial, bionic(default), focal"
    echo "                          If BuildArch is armel, this can be tizen(default)"
    echo "--setproxy=IP - optional, IP is the proxy server IP address or url with portnumber"
    echo "                           default no proxy. Example: --setproxy=127.1.2.3:8080"
    echo "--skipunmount - optional, will skip the unmount of rootfs folder."
    exit 1
}

__CrossDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
__InitialDir=$PWD
__UbuntuRepo="http://ports.ubuntu.com/"

__BuildArch=arm
__QemuArch=armhf
__LinuxCodeName=bionic
__SkipUnmount=0
__IsProxySet=0
__Apt=""
# base development support
__UbuntuPackages="build-essential"

# other development supports
__UbuntuPackages+=" ocl-icd-opencl-dev"
__UbuntuPackages+=" libhdf5-dev"
__UbuntuBoostPackages=" libboost-all-dev"

# symlinks fixer
__UbuntuPackages+=" symlinks"

__UnprocessedBuildArgs=

for i in "$@" ; do
    lowerI="$(echo $i | awk '{print tolower($0)}')"
    case $lowerI in
        -?|-h|--help)
            usage
            exit 1
            ;;
        arm)
            __BuildArch=arm
            __QemuArch=armhf
            ;;
        aarch64)
            __BuildArch=aarch64
            __QemuArch=arm64
            ;;
        armel)
            __BuildArch=armel
            __Tizen=tizen
            __QemuArch=
            __UbuntuRepo=
            __LinuxCodeName=
            ;;
        tizen)
            if [ "$__BuildArch" != "armel" ]; then
                echo "Tizen rootfs is available only for armel."
                usage;
                exit 1;
            fi
            __Tizen=tizen
            __QemuArch=
            __UbuntuRepo=
            __LinuxCodeName=
            ;;
        xenial)
            __LinuxCodeName=xenial
            ;;
        bionic)
            __LinuxCodeName=bionic
            ;;
        focal)
            __LinuxCodeName=focal
            __UbuntuBoostPackages=" libboost1.67-all-dev"
            ;;
        --setproxy*)
            proxyip="${i#*=}"
            __Apt="Acquire::http::proxy \"http://$proxyip/\";\n"
            __Apt+="Acquire::https::proxy \"http://$proxyip/\";\n"
            __Apt+="Acquire::ftp::proxy \"ftp://$proxyip/\";"
            __IsProxySet=1
            ;;
        --skipunmount)
            __SkipUnmount=1
            ;;
        *)
            __UnprocessedBuildArgs="$__UnprocessedBuildArgs $i"
            ;;
    esac
done

# Current runtime build system supports boost version under 1.70
__UbuntuPackages+="$__UbuntuBoostPackages"

__RootfsDir="$__CrossDir/rootfs/$__BuildArch"

if [[ -n "$ROOTFS_DIR" ]]; then
    __RootfsDir=$ROOTFS_DIR
fi

if [ -d "$__RootfsDir" ]; then
    if [ $__SkipUnmount == 0 ]; then
        umount $__RootfsDir/*
    fi
    rm -rf $__RootfsDir
fi

if [ $__IsProxySet == 1 ] && [ "$__Tizen" != "tizen" ]; then
    mkdir -p $__RootfsDir/etc/apt/apt.conf.d
    echo -e "$__Apt" >> $__RootfsDir/etc/apt/apt.conf.d/90proxy
fi

if [[ -n $__LinuxCodeName ]]; then
    qemu-debootstrap --arch $__QemuArch $__LinuxCodeName $__RootfsDir $__UbuntuRepo
    cp $__CrossDir/$__BuildArch/sources.list.$__LinuxCodeName $__RootfsDir/etc/apt/sources.list
    chroot $__RootfsDir apt-get update
    chroot $__RootfsDir apt-get -f -y install
    chroot $__RootfsDir apt-get -y install $__UbuntuPackages
    machine=$(chroot $__RootfsDir gcc -dumpmachine)
    chroot $__RootfsDir ln -s /usr/lib/${machine}/libhdf5_serial.a /usr/lib/${machine}/libhdf5.a
    chroot $__RootfsDir ln -s /usr/lib/${machine}/libhdf5_serial.so /usr/lib/${machine}/libhdf5.so
    chroot $__RootfsDir symlinks -cr /usr

    if [ $__SkipUnmount == 0 ]; then
        umount $__RootfsDir/*
    fi
elif [ "$__Tizen" == "tizen" ]; then
    ROOTFS_DIR=$__RootfsDir $__CrossDir/$__BuildArch/tizen-build-rootfs.sh
else
    echo "Unsupported target platform."
    usage;
    exit 1
fi
