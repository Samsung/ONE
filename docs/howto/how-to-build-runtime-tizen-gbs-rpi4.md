# How to Build Runtime with GBS for Tizen/RPi4

This document describes how to build runtime with GBS for Tizen AARCH64.
As a real example, we'll aslo describe how to prepare Tizen on Raspberry Pi 4
and show you how to run our test package runner `nnpackage_run`.

For ARM32, there would be not much difference with some changes.

Host PC is Ubuntu 18.04 but other versions or distro may work with a little
adjustments.

Detailed technical informations are not described here so please read referecnce
pages while you go on.


## Setting up build environment

(1) Add Tizen build tools repo
```
$ sudo vim /etc/apt/sources.list
```
Add this at the end
```
deb [trusted=yes] http://download.tizen.org/tools/latest-release/Ubuntu_18.04/ /
```
Note: There's a slash('/`) at the end.

For other versions of Ubuntu, please refer
http://download.tizen.org/tools/latest-release/ lists.

(2) Update package informations and upgrade to latest
```
$ sudo apt-get update
$ sudo apt-get upgrade
```

(3) Install GBS tools
```
$ sudo apt-get install gbs mic
```

To get more informations, please refer [HERE](https://source.tizen.org/ko/documentation/developer-guide/getting-started-guide/installing-development-tools)

## Build ONERT

(1) Set `python2` as default python

Some tools of GBS run in `python2` and won't run with `python3`.
Please check `python` version and set it to 2.x.

(2) set `TIZEN_BUILD_ROOT`

You may set `GBS-ROOT` to any place you like. Ususally we use home folder.
```
$ export TIZEN_BUILD_ROOT=$HOME/GBS-ROOT/
```
Adding to `$HOME/.profile` file would be a good thing.

(3) clone ONE repo

```
git clone https://github.com/Samsung/ONE.git
```

(4) Build

```
$ cd ONE

$ gbs -c infra/nnfw/config/gbs.conf build --include-all -A aarch64 --define 'test_build 1'
```
- `-A aarch64` is to set architecture to AARCH64. Use `arm32` for ARM32 target.
- `--define 'test_build 1'` is to enable test build so that we can use `nnpackage_run`

Now take a cup of coffee.

(5) Build result RPM packages

```
$ ls ~/GBS-ROOT/local/repos/tizen/aarch64/RPMS
nnfw-1.10.0-1.aarch64.rpm
nnfw-debuginfo-1.10.0-1.aarch64.rpm
nnfw-debugsource-1.10.0-1.aarch64.rpm
nnfw-devel-1.10.0-1.aarch64.rpm
nnfw-minimal-app-1.10.0-1.aarch64.rpm
nnfw-minimal-app-debuginfo-1.10.0-1.aarch64.rpm
nnfw-plugin-devel-1.10.0-1.aarch64.rpm
nnfw-test-1.10.0-1.aarch64.rpm
nnfw-test-debuginfo-1.10.0-1.aarch64.rpm
```

`-1.10.0-1` may differ as this document was written with under `1.10.0` development.

## Prepare Tizen on Raspberry Pi 4

Please refer https://wiki.tizen.org/Quick_guide_for_RPI4 for detailed descriptions.

(1) Download flashing tool
```
$ wget \
https://git.tizen.org/cgit/platform/kernel/u-boot/plain/scripts/tizen/sd_fusing_rpi3.sh?h=tizen \
--output-document=sd_fusing_rpi3.sh

$ chmod 755 sd_fusing_rpi3.sh
```

(2) Prepare Micro-SD memory card.

You first need to find out device name. This document will skip how to find this.
Suppose it's `/dev/sdj`:
```
$ sudo ./sd_fusing_rpi3.sh -d /dev/sdj --format
```
You need to change `/dev/sdj` to your configuration.

Partition table may look like this
```
Device     Boot    Start      End  Sectors  Size Id Type
/dev/sdj1  *        8192   139263   131072   64M  e W95 FAT16 (LBA)
/dev/sdj2         139264  6430719  6291456    3G 83 Linux
/dev/sdj3        6430720  9183231  2752512  1.3G 83 Linux
/dev/sdj4        9183232 62521343 53338112 25.4G  5 Extended
/dev/sdj5        9185280 61958143 52772864 25.2G 83 Linux
/dev/sdj6       61960192 62025727    65536   32M 83 Linux
/dev/sdj7       62027776 62044159    16384    8M 83 Linux
/dev/sdj8       62046208 62111743    65536   32M 83 Linux
/dev/sdj9       62113792 62130175    16384    8M 83 Linux
/dev/sdj10      62132224 62263295   131072   64M 83 Linux
/dev/sdj11      62265344 62521343   256000  125M 83 Linux
```

(3) Download images

Please visit http://download.tizen.org/snapshots/tizen/unified/latest/images/standard/iot-boot-arm64-rpi4/
and http://download.tizen.org/snapshots/tizen/unified/latest/images/standard/iot-headed-3parts-aarch64-rpi.

Pleae Visit `iot-boot-armv7l-rpi4` folder for ARM32 images.

Get latest file. As of writing this document, name has `20200908.3`.
```
$ wget  http://download.tizen.org/snapshots/tizen/unified/latest/images/standard/iot-boot-arm64-rpi4/tizen-unified_20200908.3_iot-boot-arm64-rpi4.tar.gz

$ wget http://download.tizen.org/snapshots/tizen/unified/latest/images/standard/iot-headed-3parts-aarch64-rpi/tizen-unified_20200908.3_iot-headed-3parts-aarch64-rpi.tar.gz
```

(4) Flash images to memory card

As like above, suppose memory card is at `/dev/sdj`
```
$ sudo ./sd_fusing_rpi3.sh -d /dev/sdj \
-b tizen-unified_20200908.3_iot-boot-arm64-rpi4.tar.gz \
tizen-unified_20200908.3_iot-headed-3parts-aarch64-rpi.tar.gz
```
You need to change `/dev/sdj` to your configuration and also `tizen-unified_...` file to your
latest download file name.

(5) Assign IP address for `sdb` connection

Here, we provide a way to connect `sdb` tool through TCP/IP.

Below steps will modify root image and set fixed IP address.

(5-1) Mount image to host
```
$ mkdir j2
$ sudo mount /dev/sdj2 j2
```
As like above, please update `/dev/sdj2` to your configuration.

Add a new file
```
$ vi j2/etc/systemd/system/ip.service
```
and set as like:
```
[Service]
Type=simple
Restart=always
RestartSec=1
User=root
ExecStart=/bin/sh /bin/ip.sh

[Install]
WantedBy=multi-user.target
```

Add a new file
```
$ vi j2/bin/ip.sh
```
and set with IP address for your RPi4:
```
ifconfig eth0 192.168.x.y netmask 255.255.255.0 up
```
where you should update `192.168.x.y` part to your actual IP address.


Add a symbolic link
```
$ pushd j2/etc/systemd/system/multi-user.target.wants/
$ sudo ln -s ../../system/ip.service .
$ popd
```
Now that every thing is ready, unmount and unplug your memory card and plug into
RPi4, turn on the power.
```
$ sync
$ sudo umount j2
```

## sdb connect to Tizen/RPi4

You may need to install Tizen Studio to use `sdb` command.
Please visit https://developer.tizen.org/ if you don't have this.

We assume `sdb` command is in the PATH.

(1) Connect

```
$ sdb connect 192.168.x.y
connecting to 192.168.x.y:26101 ...
connected to 192.168.x.y:26101
```
Please update `192.168.x.y` part to your actual IP address.

Check with `devices` command: you should see `rpi3` or alike.
```
$ sdb devices
List of devices attached
192.168.x.y:26101     device          rpi3
```

(2) Remount filesystem with R/W

You need to remount file system with Read/Write so that you can install packages.
```
$ sdb root on
$ sdb shell
```
Inside your Tizen/RPi4:
```
sh-3.2# mount -o rw,remount /
```

(3) Download dependent packages

In your host, maybe with another terminal, download packages from
http://download.tizen.org/releases/daily/tizen/unified/latest/repos/standard/packages/aarch64/

```
$ wget http://download.tizen.org/releases/daily/tizen/unified/latest/repos/standard/packages/aarch64/libarmcl-v20.05-17.5.aarch64.rpm

$ wget http://download.tizen.org/releases/daily/tizen/unified/latest/repos/standard/packages/aarch64/libhdf5-101-1.10.1-3.85.aarch64.rpm

$ wget http://download.tizen.org/releases/daily/tizen/unified/latest/repos/standard/packages/aarch64/libhdf5_cpp101-1.10.1-3.85.aarch64.rpm
```

(4) Copy to device
```
$ sdb push libarmcl-v20.05-17.5.aarch64.rpm /opt/usr/home/owner/share/tmp/
$ sdb push libhdf5-101-1.10.1-3.85.aarch64.rpm /opt/usr/home/owner/share/tmp/
$ sdb push libhdf5_cpp101-1.10.1-3.85.aarch64.rpm /opt/usr/home/owner/share/tmp/
```
And our runtime packages
```
$ cd ~/GBS-ROOT/local/repos/tizen/aarch64/RPMS
$ sdb push nnfw-1.10.0-1.aarch64.rpm /opt/usr/home/owner/share/tmp/
$ sdb push nnfw-test-1.10.0-1.aarch64.rpm /opt/usr/home/owner/share/tmp/
```

(5) Install dependent packages

Within Tizen/RPi4 shell
```
sh-3.2# cd /opt/usr/home/owner/share/tmp/

sh-3.2# rpm -i libarmcl-v20.05-17.5.aarch64.rpm
sh-3.2# rpm -i libhdf5-101-1.10.1-3.85.aarch64.rpm
sh-3.2# rpm -i libhdf5_cpp101-1.10.1-3.85.aarch64.rpm
```
There may be message like this but it seems OK:
```
/sbin/ldconfig: Cannot lstat /lib64/libhdf5.so.101.0.0: Permission denied
```
Continue install
```
sh-3.2# rpm -i nnfw-1.10.0-1.aarch64.rpm
sh-3.2# rpm -i nnfw-test-1.10.0-1.aarch64.rpm
```

Our `Product` binary folder is installed at `/opt/usr/nnfw-test`.
```
sh-3.2# cd /opt/usr/nnfw-test
sh-3.2# ls -al
total 16
drwxr-xr-x  4 root root 4096 Jan  1 09:05 .
drwxr-xr-x 14 root root 4096 Jan  1 09:05 ..
drwxr-xr-x  3 root root 4096 Jan  1 09:05 Product
drwxr-xr-x  3 root root 4096 Jan  1 09:05 infra
```

(6) Run nnpackage

Refer `how-to-build-package.md` document to produce nnpackage from a model.

Assume `mobilenet_v2_1.4_224` nnpackage is already copied to
`/opt/usr/home/owner/media/models` folder with `sdb` command.

```
sh-3.2# BACKENDS="cpu" Product/out/bin/nnpackage_run \
--nnpackage /opt/usr/home/owner/media/models/mobilenet_v2_1.4_224

Package Filename /opt/usr/home/owner/media/models/mobilenet_v2_1.4_224
===================================
MODEL_LOAD   takes 65.403 ms
PREPARE      takes 158.716 ms
EXECUTE      takes 373.447 ms
- MEAN     :  373.447 ms
- MAX      :  373.447 ms
- MIN      :  373.447 ms
- GEOMEAN  :  373.447 ms
===================================
```
