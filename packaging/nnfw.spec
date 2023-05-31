Name:    nnfw
Summary: nnfw
Version: 1.23.0
Release: 1
Group:   Development
License: Apache-2.0 and MIT and BSD-2-Clause and MPL-2.0

Source0: %{name}-%{version}.tar.gz
Source1: %{name}.manifest
Source1001: nnapi_test_generated.tar.gz
Source2001: nnfw.pc.in
Source2002: nnfw-plugin.pc.in
Source3001: ABSEIL.tar.gz
Source3002: CPUINFO.tar.gz
Source3003: EGL_HEADERS.tar.gz
Source3004: FARMHASH.tar.gz
Source3005: FP16.tar.gz
Source3006: FXDIV.tar.gz
Source3007: GEMMLOWP.tar.gz
Source3008: OOURAFFT.tar.gz
Source3009: OPENCL_HEADERS.tar.gz
Source3010: OPENGL_HEADERS.tar.gz
Source3011: PSIMD.tar.gz
Source3012: PTHREADPOOL.tar.gz
Source3013: TENSORFLOW-2.8.0-EIGEN.tar.gz
Source3014: TENSORFLOW-2.8.0-GEMMLOWP.tar.gz
Source3015: TENSORFLOW-2.8.0-RUY.tar.gz
Source3016: TENSORFLOW-2.8.0.tar.gz
Source3017: VULKAN.tar.gz
Source3018: XNNPACK.tar.gz
Source3019: FLATBUFFERS-2.0.tar.gz
Source3020: jsoncpp.tar.gz

%{!?build_type:     %define build_type      Release}
%{!?npud_build:     %define npud_build      1}
%{!?trix_support:   %define trix_support    1}
%{!?coverage_build: %define coverage_build  0}
%{!?test_build:     %define test_build      0}
%{!?extra_option:   %define extra_option    %{nil}}
%{!?config_support: %define config_support  1}

%if %{coverage_build} == 1
# Coverage test requires debug build runtime
%define build_type Debug
%define test_build 1
%endif

BuildRequires:  cmake

Requires(post): /sbin/ldconfig
Requires(postun): /sbin/ldconfig

%if %{test_build} == 1
BuildRequires:  pkgconfig(boost)
BuildRequires:  pkgconfig(tensorflow2-lite)
BuildRequires:  hdf5-devel
BuildRequires:  libaec-devel
BuildRequires:  pkgconfig(zlib)
BuildRequires:  pkgconfig(libjpeg)
BuildRequires:  gtest-devel
%endif

%if %{npud_build} == 1
BuildRequires:  pkgconfig(glib-2.0)
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

%package minimal-app
Summary: Minimal test binary for VD manual test

%description minimal-app
Minimal test binary for VD manual test

%if %{test_build} == 1
%package test
Summary: NNFW Test

%description test
NNFW test rpm.
If you want to use test package, you should install runtime package which is build with test build option
If you want to get coverage info, you should install runtime package which is build with coverage build option
# TODO Use release runtime pacakge for test
%endif

%if %{npud_build} == 1
%package npud
Summary: NPU daemon

%description npud
NPU daemon for optimal management of NPU hardware
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

%define install_dir %{_prefix}
%define install_path %{buildroot}%{install_dir}
%define nnfw_workspace build
%define build_env NNFW_WORKSPACE=%{nnfw_workspace}

# Path to install test bin and scripts (test script assumes path Product/out)
# TODO Share path with release package
%define test_install_home /opt/usr/nnfw-test
%define test_install_dir %{test_install_home}/Product/out
%define test_install_path %{buildroot}%{test_install_dir}

# Set option for test build (and coverage test build)
%define option_test -DENABLE_TEST=OFF
%define option_coverage %{nil}
%define test_suite_list infra/scripts tests/scripts

%if %{test_build} == 1
# ENVVAR_ONERT_CONFIG: Use environment variable for runtime core configuration and debug
%define option_test -DENABLE_TEST=ON -DENVVAR_ONERT_CONFIG=ON
%endif # test_build

# Set option for configuration
%define option_config %{nil}
%if %{config_support} == 1
%if %{npud_build} == 1
# ENVVAR_NPUD_CONFIG: Use environment variable for npud configuration and debug
%define option_config -DENVVAR_NPUD_CONFIG=ON
%endif # npud_build
%endif # config_support

%if %{coverage_build} == 1
%define option_coverage -DENABLE_COVERAGE=ON
%endif # coverage_build

%define build_options -DCMAKE_BUILD_TYPE=%{build_type} -DTARGET_ARCH=%{target_arch} -DTARGET_OS=tizen -DBUILD_MINIMAL_SAMPLE=ON \\\
        %{option_test} %{option_coverage} %{option_config} %{extra_option}

%prep
%setup -q
cp %{SOURCE1} .
mkdir ./externals
tar -xf %{SOURCE1001} -C ./tests/nnapi/src/
tar -xf %{SOURCE3001} -C ./externals
tar -xf %{SOURCE3002} -C ./externals
tar -xf %{SOURCE3003} -C ./externals
tar -xf %{SOURCE3004} -C ./externals
tar -xf %{SOURCE3005} -C ./externals
tar -xf %{SOURCE3006} -C ./externals
tar -xf %{SOURCE3007} -C ./externals
tar -xf %{SOURCE3008} -C ./externals
tar -xf %{SOURCE3009} -C ./externals
tar -xf %{SOURCE3010} -C ./externals
tar -xf %{SOURCE3011} -C ./externals
tar -xf %{SOURCE3012} -C ./externals
tar -xf %{SOURCE3013} -C ./externals
tar -xf %{SOURCE3014} -C ./externals
tar -xf %{SOURCE3015} -C ./externals
tar -xf %{SOURCE3016} -C ./externals
tar -xf %{SOURCE3017} -C ./externals
tar -xf %{SOURCE3018} -C ./externals
tar -xf %{SOURCE3019} -C ./externals
tar -xf %{SOURCE3020} -C ./externals

%build
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86
# compiler build
make -f infra/nncc/Makefile.arm32.native cfg
make -f infra/nncc/Makefile.arm32.native debug
make -f infra/nncc/Makefile.arm32.native test

# runtime build
%{build_env} ./nnfw configure %{build_options}
%{build_env} ./nnfw build -j4
# install in workspace
# TODO Set install path
%{build_env} ./nnfw install

%if %{test_build} == 1
%if %{coverage_build} == 1
pwd > tests/scripts/build_path.txt
%endif # coverage_build
tar -zcf test-suite.tar.gz infra/scripts
%endif # test_build
%endif # arm armv7l armv7hl aarch64

%install
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86

mkdir -p %{buildroot}%{_libdir}
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_includedir}
install -m 644 build/out/lib/*.so %{buildroot}%{_libdir}
install -m 755 build/out/bin/onert-minimal-app %{buildroot}%{_bindir}
cp -r build/out/include/* %{buildroot}%{_includedir}/

# For developer
cp %{SOURCE2001} .
cp %{SOURCE2002} .
sed -i 's:@libdir@:%{_libdir}:g
        s:@includedir@:%{_includedir}:g
        s:@version@:%{version}:g' ./nnfw.pc.in
sed -i 's:@libdir@:%{_libdir}:g
        s:@includedir@:%{_includedir}:g
        s:@version@:%{version}:g' ./nnfw-plugin.pc.in
mkdir -p %{buildroot}%{_libdir}/pkgconfig
install -m 0644 ./nnfw.pc.in %{buildroot}%{_libdir}/pkgconfig/nnfw.pc
install -m 0644 ./nnfw-plugin.pc.in %{buildroot}%{_libdir}/pkgconfig/nnfw-plugin.pc

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
cp tests/nnapi/nnapi_gtest.skip.%{target_arch}-* %{test_install_path}/nnapi-gtest/.
cp %{test_install_path}/nnapi-gtest/nnapi_gtest.skip.%{target_arch}-linux.cpu %{test_install_path}/nnapi-gtest/nnapi_gtest.skip
tar -zxf test-suite.tar.gz -C %{buildroot}%{test_install_home}

%if %{coverage_build} == 1
mkdir -p %{buildroot}%{test_install_home}/gcov
find %{nnfw_workspace} -name "*.gcno" -exec xargs cp {} %{buildroot}%{test_install_home}/gcov/. \;
install -m 0644 ./tests/scripts/build_path.txt %{buildroot}%{test_install_dir}/test/build_path.txt
%endif # coverage_build
%endif # test_build

%if %{npud_build} == 1
install -m 755 build/out/bin/npud %{buildroot}%{_bindir}

%if %{test_build} == 1
mkdir -p %{test_install_path}/npud-gtest
install -m 755 build/out/npud-gtest/* %{test_install_path}/npud-gtest
%endif # test_build

%endif # npud_build

%endif

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86
%{_libdir}/*.so
%exclude %{_includedir}/CL/*
%endif

%files devel
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86
%dir %{_includedir}/nnfw
%{_includedir}/nnfw/*
%{_libdir}/pkgconfig/nnfw.pc
%endif

%files plugin-devel
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86
%dir %{_includedir}/onert
%{_includedir}/onert/*
%{_libdir}/pkgconfig/nnfw-plugin.pc
%endif

%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86
%files minimal-app
%manifest %{name}.manifest
%defattr(-,root,root,-)
%{_bindir}/onert-minimal-app
%endif

%if %{test_build} == 1
%files test
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64
%dir %{test_install_home}
%{test_install_home}/*
%endif # arm armv7l armv7hl aarch64
%endif # test_build

%if %{npud_build} == 1
%files npud
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l armv7hl aarch64 x86_64 %ix86
%{_bindir}/npud
%endif # arm armv7l armv7hl aarch64 x86_64 %ix86
%endif # npud_build

%changelog
* Thu Mar 15 2018 Chunseok Lee <chunseok.lee@samsung.com>
- Initial spec file for nnfw
