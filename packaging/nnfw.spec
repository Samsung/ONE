Name:    nnfw
Summary: nnfw
Version: 1.16.0
Release: 1
Group:   Development
License: Apache-2.0 and MIT and BSD-2-Clause

Source0: %{name}-%{version}.tar.gz
Source1: %{name}.manifest
Source1001: nnapi_test_generated.tar.gz
Source1002: gtest.tar.gz
Source1003: eigen.tar.gz
Source1004: gemmlowp.tar.gz
Source1005: ruy.tar.gz
Source1006: cpuinfo.tar.gz
Source1007: XNNPACK.tar.gz
Source1008: FXDIV.tar.gz
Source1009: PTHREADPOOL.tar.gz
Source1010: PSIMD.tar.gz
Source1011: FP16.tar.gz
Source2001: nnfw.pc.in
Source2002: nnfw-plugin.pc.in

%{!?build_type:     %define build_type      Release}
%{!?coverage_build: %define coverage_build  0}
%{!?test_build:     %define test_build      0}
%{!?extra_option:   %define extra_option    %{nil}}
%if %{coverage_build} == 1
%define test_build 1
%endif

BuildRequires:  cmake
# Require flatbuffers-devel for onert frontend (model loading)
BuildRequires:  flatbuffers-devel
BuildRequires:  abseil-cpp-devel

%ifarch %{arm} aarch64
# Require python for acl-ex library build pre-process
BuildRequires:  python
BuildRequires:  libarmcl-devel >= v20.05
%endif

Requires(post): /sbin/ldconfig
Requires(postun): /sbin/ldconfig

%if %{test_build} == 1
BuildRequires:  boost-devel
BuildRequires:  tensorflow-lite-devel
BuildRequires:  hdf5-devel
BuildRequires:  libaec-devel
BuildRequires:  zlib-devel
BuildRequires:  libjpeg-devel
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
NNFW test rpm. It does not depends on nnfw rpm since it contains nnfw runtime.
%endif

%ifarch %{arm}
%define target_arch armv7l
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
%define build_env NNFW_WORKSPACE=build
%define build_options -DCMAKE_BUILD_TYPE=%{build_type} -DTARGET_ARCH=%{target_arch} -DTARGET_OS=tizen -DENABLE_TEST=off -DBUILD_MINIMAL_SAMPLE=on

# Set option for test build (and coverage test build)
%define test_install_home /opt/usr/nnfw-test
%define test_install_dir %{test_install_home}/Product/out
%define test_install_path %{buildroot}%{test_install_dir}
%define coverage_option %{nil}
%define test_suite_list infra/scripts tests/scripts
%define test_build_type %{build_type}
%if %{coverage_build} == 1
%define coverage_option -DENABLE_COVERAGE=ON
%define test_build_type Debug
%endif
%define test_build_env NNFW_INSTALL_PREFIX=%{test_install_path} NNFW_WORKSPACE=build_for_test
%define test_build_options %{coverage_option} -DCMAKE_BUILD_TYPE=%{test_build_type} -DTARGET_ARCH=%{target_arch} -DTARGET_OS=tizen -DENVVAR_ONERT_CONFIG=ON

%prep
%setup -q
cp %{SOURCE1} .
mkdir ./externals
tar -xf %{SOURCE1001} -C ./tests/nnapi/src/
tar -xf %{SOURCE1002} -C ./externals
tar -xf %{SOURCE1003} -C ./externals
tar -xf %{SOURCE1004} -C ./externals
tar -xf %{SOURCE1005} -C ./externals
tar -xf %{SOURCE1006} -C ./externals
tar -xf %{SOURCE1007} -C ./externals
tar -xf %{SOURCE1008} -C ./externals
tar -xf %{SOURCE1009} -C ./externals
tar -xf %{SOURCE1010} -C ./externals
tar -xf %{SOURCE1011} -C ./externals

%build
%ifarch arm armv7l aarch64 x86_64 %ix86
# runtime build
%{build_env} ./nnfw configure %{build_options} %{extra_option}
%{build_env} ./nnfw build -j4
# install in workspace
# TODO Set install path
%{build_env} ./nnfw install

%if %{test_build} == 1
# test runtime
# TODO remove duplicated build process
%{test_build_env} ./nnfw configure %{test_build_options} %{extra_option}
%{test_build_env} ./nnfw build -j4
%if %{coverage_build} == 1
pwd > tests/scripts/build_path.txt
%endif # coverage_build
tar -zcf test-suite.tar.gz infra/scripts
%endif # test_build
%endif # arm armv7l aarch64

%install
%ifarch arm armv7l aarch64 x86_64 %ix86

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
%{test_build_env} ./nnfw install
# Share test script with ubuntu (ignore error if there is no list for target)
cp tests/nnapi/nnapi_gtest.skip.%{target_arch}-* %{buildroot}%{test_install_dir}/unittest/.
cp %{buildroot}%{test_install_dir}/unittest/nnapi_gtest.skip.%{target_arch}-linux.cpu %{buildroot}%{test_install_dir}/unittest/nnapi_gtest.skip
tar -zxf test-suite.tar.gz -C %{buildroot}%{test_install_home}

%if %{coverage_build} == 1
mkdir -p %{buildroot}%{test_install_home}/gcov
find . -name "*.gcno" -exec xargs cp {} %{buildroot}%{test_install_home}/gcov/. \;
install -m 0644 ./tests/scripts/build_path.txt %{buildroot}%{test_install_dir}/test/build_path.txt
%endif # coverage_build
%endif # test_build

%endif

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l aarch64 x86_64 %ix86
%{_libdir}/*.so
%endif

%files devel
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l aarch64 x86_64 %ix86
%dir %{_includedir}/nnfw
%{_includedir}/nnfw/*
%{_libdir}/pkgconfig/nnfw.pc
%endif

%files plugin-devel
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l aarch64 x86_64 %ix86
%dir %{_includedir}/onert
%{_includedir}/onert/*
%{_libdir}/pkgconfig/nnfw-plugin.pc
%endif

%ifarch arm armv7l aarch64 x86_64 %ix86
%files minimal-app
%manifest %{name}.manifest
%defattr(-,root,root,-)
%{_bindir}/onert-minimal-app
%endif

%if %{test_build} == 1
%files test
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l aarch64 x86_64
%dir %{test_install_home}
%{test_install_home}/*
%endif # arm armv7l aarch64
%endif # test_build

%changelog
* Thu Mar 15 2018 Chunseok Lee <chunseok.lee@samsung.com>
- Initial spec file for nnfw
