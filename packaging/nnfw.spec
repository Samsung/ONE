Name:    nnfw
Summary: nnfw
Version: 1.31.0
Release: 1
Group:   Development
License: Apache-2.0 and MIT and BSD-2-Clause and MPL-2.0

Source0: %{name}-%{version}.tar.gz
Source1: %{name}.manifest
Source1001: nnapi_test_generated.tar.gz
Source2001: onert.pc.in
Source2002: onert-plugin.pc.in
Source3001: ABSEIL.tar.gz
Source3002: CPUINFO.tar.gz
Source3003: FARMHASH.tar.gz
Source3004: FLATBUFFERS-23.5.26.tar.gz
Source3005: FP16.tar.gz
Source3006: FXDIV.tar.gz
Source3007: MLDTYPES.tar.gz
Source3008: NEON2SSE.tar.gz
Source3009: OOURAFFT.tar.gz
Source3010: PSIMD.tar.gz
Source3011: PTHREADPOOL.tar.gz
Source3012: TENSORFLOW-2.16.1-EIGEN.tar.gz
Source3013: TENSORFLOW-2.16.1-GEMMLOWP.tar.gz
Source3014: TENSORFLOW-2.16.1-RUY.tar.gz
Source3015: TENSORFLOW-2.16.1.tar.gz
Source3016: XNNPACK.tar.gz

%{!?build_type:     %define build_type      Release}
%{!?trix_support:   %define trix_support    1}
%{!?odc_build:      %define odc_build       1}
%{!?test_build:     %define test_build      0}
%{!?extra_option:   %define extra_option    %{nil}}
# Define nproc on gbs build option if you want to set number of build threads manually (ex. CI/CD infra)
%define build_jobs   %{?!nproc:%{?_smp_mflags}%{?!_smp_mflags:-j4}}%{?nproc:-j%nproc}
%{!?nproc:          %define nproc           %{?!jobs:4}%{?jobs}}

BuildRequires:  cmake

Requires(post): /sbin/ldconfig
Requires(postun): /sbin/ldconfig

%if %{test_build} == 1
BuildRequires:  pkgconfig(tensorflow2-lite)
BuildRequires:  hdf5-devel-static
BuildRequires:  libaec-devel
BuildRequires:  pkgconfig(zlib)
BuildRequires:  pkgconfig(libjpeg)
BuildRequires:  gtest-devel
%endif

%if %{trix_support} == 1
BuildRequires:  pkgconfig(npu-engine)
%endif

%description
nnfw is a high-performance, on-device neural network framework for Tizen

%package devel
Summary: NNFW Devel Package
Requires: %{name} = %{version}-%{release}

%description devel
NNFW development package for application developer using runtime

%package plugin-devel
Summary: NNFW Devel Package
Requires: %{name}-devel = %{version}-%{release}

%description plugin-devel
NNFW development package for backend plugin developer

%package train
Summary: ONERT Train Package
Requires: %{name} = %{version}-%{release}

%description train
ONERT backend package for training

%if %{trix_support} == 1
%package trix
Summary: ONERT Trix Package
Requires: %{name} = %{version}-%{release}

%description trix
ONERT loader and backend package for trix
%endif # trix_support

%if %{odc_build} == 1
%package odc
Summary: NNFW On-Device Compilation Package

%description odc
NNFW package for on-device compilation
%endif # odc_build

%if %{test_build} == 1
%package test
Summary: NNFW Test

%description test
NNFW test rpm.
If you want to use test package, you should install runtime package which is build with test build option
# TODO Use release runtime pacakge for test
%endif

%ifarch armv7l
%define target_arch armv7l
%endif
%ifarch armv7hl
%define target_arch armv7hl
%endif
%ifarch x86_64
%define target_arch x86_64
%endif
%ifarch aarch64
%define target_arch aarch64
%endif
%ifarch %ix86
%define target_arch i686
%endif
%ifarch riscv64
%define target_arch riscv64
%endif

%define install_dir %{_prefix}
%define install_path %{buildroot}%{install_dir}
%define nnfw_workspace build
%define build_env NNFW_WORKSPACE=%{nnfw_workspace}
%define nncc_workspace build/nncc
%define nncc_env NNCC_WORKSPACE=%{nncc_workspace}
%define overlay_path %{nnfw_workspace}/overlay

# Path to install test bin and scripts (test script assumes path Product/out)
# TODO Share path with release package
%define test_install_home /opt/usr/nnfw-test
%define test_install_dir %{test_install_home}/Product/out
%define test_install_path %{buildroot}/%{test_install_dir}

# Set option for test build
%define option_test -DENABLE_TEST=OFF
%define test_suite_list infra/scripts tests/scripts

%if %{test_build} == 1
%define option_test -DENABLE_TEST=ON
%endif # test_build

# Set option for configuration
%define option_config %{nil}

%define build_options -DCMAKE_BUILD_TYPE=%{build_type} -DTARGET_ARCH=%{target_arch} -DTARGET_OS=tizen \\\
        -DEXTERNALS_BUILD_THREAD=%{nproc} -DBUILD_MINIMAL_SAMPLE=OFF -DNNFW_OVERLAY_DIR=$(pwd)/%{overlay_path} \\\
        %{option_test} %{option_config} %{extra_option}

%define strip_options %{nil}
%if %{build_type} == "Release"
%define strip_options --strip
%endif

%prep
%setup -q
cp %{SOURCE1} .
mkdir ./runtime/externals
tar -xf %{SOURCE1001} -C ./runtime/tests/nnapi/src/
tar -xf %{SOURCE3001} -C ./runtime/externals
tar -xf %{SOURCE3002} -C ./runtime/externals
tar -xf %{SOURCE3003} -C ./runtime/externals
tar -xf %{SOURCE3004} -C ./runtime/externals
tar -xf %{SOURCE3005} -C ./runtime/externals
tar -xf %{SOURCE3006} -C ./runtime/externals
tar -xf %{SOURCE3007} -C ./runtime/externals
tar -xf %{SOURCE3008} -C ./runtime/externals
tar -xf %{SOURCE3009} -C ./runtime/externals
tar -xf %{SOURCE3010} -C ./runtime/externals
tar -xf %{SOURCE3011} -C ./runtime/externals
tar -xf %{SOURCE3012} -C ./runtime/externals
tar -xf %{SOURCE3013} -C ./runtime/externals
tar -xf %{SOURCE3014} -C ./runtime/externals
tar -xf %{SOURCE3015} -C ./runtime/externals
tar -xf %{SOURCE3016} -C ./runtime/externals

%if %{odc_build} == 1
mkdir ./externals
tar -xf %{SOURCE3004} -C ./externals
tar -xf %{SOURCE3005} -C ./externals
tar -xf %{SOURCE3008} -C ./externals
tar -xf %{SOURCE3013} -C ./externals
tar -xf %{SOURCE3014} -C ./externals
tar -xf %{SOURCE3015} -C ./externals
%endif # odc_build

%build
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
# nncc build
%if %{odc_build} == 1
%{nncc_env} ./nncc configure -DBUILD_GTEST=OFF -DENABLE_TEST=OFF -DEXTERNALS_BUILD_THREADS=%{nproc} -DCMAKE_BUILD_TYPE=%{build_type} -DTARGET_ARCH=%{target_arch} -DTARGET_OS=tizen \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/%{overlay_path} \
	-DBUILD_WHITELIST="luci;foder;pepper-csv2vec;loco;locop;logo;logo-core;mio-circle08;luci-compute;oops;hermes;hermes-std;angkor;pp;pepper-strcast;pepper-str"
%{nncc_env} ./nncc build %{build_jobs}
cmake --install %{nncc_workspace} %{strip_options}

# install angkor TensorIndex and oops InternalExn header (TODO: Remove this)
mkdir -p %{overlay_path}/include/nncc/core/ADT/tensor
mkdir -p %{overlay_path}/include/oops
mkdir -p %{overlay_path}/include/luci/IR
mkdir -p %{overlay_path}/include/mio/circle
cp compiler/angkor/include/nncc/core/ADT/tensor/Index.h %{overlay_path}/include/nncc/core/ADT/tensor
cp compiler/oops/include/oops/InternalExn.h %{overlay_path}/include/oops
cp compiler/luci/lang/include/luci/IR/CircleNodes.lst %{overlay_path}/include/luci/IR
cp %{nncc_workspace}/compiler/mio-circle08/gen/mio/circle/schema_generated.h %{overlay_path}/include/mio/circle
cp -r %{nncc_workspace}/overlay/include/flatbuffers %{overlay_path}/include
%endif # odc_build

# runtime build
%{build_env} ./nnfw configure %{build_options}
%{build_env} ./nnfw build %{build_jobs}
# install in workspace
# TODO Set install path
%{build_env} ./nnfw install --prefix %{nnfw_workspace}/out %{strip_options}

%if %{test_build} == 1
tar -zcf test-suite.tar.gz infra/scripts
%endif # test_build
%endif # arm armv7l armv7hl aarch64

%install
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64

mkdir -p %{buildroot}%{_libdir}/nnfw/backend
mkdir -p %{buildroot}%{_libdir}/nnfw/loader
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_includedir}
install -m 644 build/out/lib/*.so %{buildroot}%{_libdir}
install -m 644 build/out/lib/nnfw/*.so %{buildroot}%{_libdir}/nnfw/
%if "%{asan}" == "1"
 install -m 644 build/out/lib/nnfw/backend/*.so %{buildroot}%{_libdir}
 install -m 644 build/out/lib/nnfw/loader/*.so %{buildroot}%{_libdir}
%else
 install -m 644 build/out/lib/nnfw/backend/*.so %{buildroot}%{_libdir}/nnfw/backend
 install -m 644 build/out/lib/nnfw/loader/*.so %{buildroot}%{_libdir}/nnfw/loader
%endif
cp -r build/out/include/* %{buildroot}%{_includedir}/

# For developer
cp %{SOURCE2001} .
cp %{SOURCE2002} .
sed -i 's:@libdir@:%{_libdir}:g
        s:@includedir@:%{_includedir}:g
        s:@version@:%{version}:g' ./onert.pc.in
sed -i 's:@libdir@:%{_libdir}:g
        s:@includedir@:%{_includedir}:g
        s:@version@:%{version}:g' ./onert-plugin.pc.in
mkdir -p %{buildroot}%{_libdir}/pkgconfig
install -m 0644 ./onert.pc.in %{buildroot}%{_libdir}/pkgconfig/onert.pc
install -m 0644 ./onert-plugin.pc.in %{buildroot}%{_libdir}/pkgconfig/onert-plugin.pc
pushd %{buildroot}%{_libdir}/pkgconfig
ln -sf onert.pc nnfw.pc
ln -sf onert-plugin.pc nnfw-plugin.pc
popd

%if %{test_build} == 1
mkdir -p %{test_install_path}/bin
mkdir -p %{test_install_path}/nnapi-gtest
mkdir -p %{test_install_path}/unittest
mkdir -p %{test_install_path}/test

install -m 755 build/out/bin/onert_run %{test_install_path}/bin
install -m 755 build/out/bin/tflite_comparator %{test_install_path}/bin
install -m 755 build/out/bin/tflite_run %{test_install_path}/bin
install -m 755 build/out/nnapi-gtest/* %{test_install_path}/nnapi-gtest
install -m 755 build/out/unittest/*_test %{test_install_path}/unittest
install -m 755 build/out/unittest/test_* %{test_install_path}/unittest
cp -r build/out/test/* %{test_install_path}/test
cp -r build/out/unittest/nnfw_api_gtest_models %{test_install_path}/unittest

# Share test script with ubuntu (ignore error if there is no list for target)
cp runtime/tests/nnapi/nnapi_gtest.skip.%{target_arch}-* %{test_install_path}/nnapi-gtest/.
cp %{test_install_path}/nnapi-gtest/nnapi_gtest.skip.%{target_arch}-linux.cpu %{test_install_path}/nnapi-gtest/nnapi_gtest.skip
tar -zxf test-suite.tar.gz -C %{buildroot}%{test_install_home}

%endif # test_build

%if %{odc_build} == 1
mkdir -p %{buildroot}%{_libdir}/nnfw/odc
install -m 644 %{overlay_path}/lib/libluci*.so %{buildroot}%{_libdir}/nnfw/odc
install -m 644 %{overlay_path}/lib/libloco*.so %{buildroot}%{_libdir}/nnfw/odc
install -m 644 build/out/lib/nnfw/odc/*.so %{buildroot}%{_libdir}/nnfw/odc
%endif # odc_build

%endif

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%{_libdir}/*.so
%{_libdir}/nnfw/*.so
%if "%{asan}" != "1"
%{_libdir}/nnfw/backend/*.so
%exclude %{_libdir}/nnfw/backend/libbackend_trix.so
%exclude %{_libdir}/nnfw/backend/libbackend_train.so
%endif
%exclude %{_includedir}/CL/*
%endif

%files devel
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%dir %{_includedir}/nnfw
%{_includedir}/nnfw/*
%{_libdir}/pkgconfig/nnfw.pc
%{_libdir}/pkgconfig/onert.pc
%endif

%files plugin-devel
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%dir %{_includedir}/onert
%{_includedir}/onert/*
%{_libdir}/pkgconfig/nnfw-plugin.pc
%{_libdir}/pkgconfig/onert-plugin.pc
%endif

%files train
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%{_libdir}/nnfw/backend/libbackend_train.so
%endif # arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64

%if %{trix_support} == 1
%files trix
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%{_libdir}/nnfw/loader/libtvn_loader.so
%{_libdir}/nnfw/backend/libbackend_trix.so
%endif # arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%endif # trix_support

%if %{test_build} == 1
%files test
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64
%dir %{test_install_home}
%{test_install_home}/*
%endif # arm armv7l armv7hl aarch64
%endif # test_build

%if %{odc_build} == 1
%files odc
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86 riscv64
%dir %{_libdir}/nnfw/odc
%{_libdir}/nnfw/odc/*
%endif # arm armv7l armv7hl aarch64 x86_64 %ix86
%endif # odc_build

%changelog
* Thu Mar 15 2018 Chunseok Lee <chunseok.lee@samsung.com>
- Initial spec file for nnfw
