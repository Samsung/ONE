Name:    TFLiteSharp
Summary: Tensorflow lite native cpp wrapper and C# API
Version: 1.0.0
Release: 1
Group:   Development/Libraries
License: Apache-2.0
Source0: %{name}-%{version}.tar.gz
Source1: %{name}.manifest
Source2: tflite-native.manifest

%description
%{summary}

%package TFLiteNative
Summary:          Tensorflow lite native cpp wrapper
Group:            Development/Libraries
BuildRequires:    cmake
BuildRequires:    pkgconfig(dlog)
BuildRequires:    pkgconfig(tensorflow-lite)
Requires(post):   /sbin/ldconfig
Requires(postun): /sbin/ldconfig

%description TFLiteNative
Native CPP Wrapper for Tensorflow lite

%package TFLiteNative-devel
Summary:  Tensorflow lite native cpp wrapper (Development)
Requires: %{name} = %{version}-%{release}

%description TFLiteNative-devel
Tensorflow lite native cpp wrapper (Development)

%package TFLiteSharp
Summary:       Tensorflow lite API for C#
Group:         Development/Libraries
AutoReqProv:   no
ExcludeArch:   aarch64

BuildRequires: dotnet-build-tools

%define Assemblies TFLiteSharp

%description TFLiteSharp
Tensorflow lite API for C#

%dotnet_import_sub_packages

%prep
%setup -q
cp %{SOURCE1} .
cp %{SOURCE2} .
%if 0%{?tizen:1}
%define TARGET_OS tizen
%else
%define TARGET_OS linux
%endif

%build
MAJORVER=`echo %{version} | awk 'BEGIN {FS="."}{print $1}'`
%if "%{TARGET_OS}" == "tizen"
cmake VERBOSE=1 -DCMAKE_INSTALL_PREFIX=/usr -DFULLVER=%{version} -DMAJORVER=${MAJORVER} \
      -DLIB_INSTALL_DIR=%{_libdir} -DINCLUDE_INSTALL_DIR=%{_includedir} \
      -DLIB_PATH=%{_lib} -DTIZEN=1 contrib/TFLiteSharp/TFLiteNative
%else
cmake VERBOSE=1 -DCMAKE_INSTALL_PREFIX=/usr -DFULLVER=%{version} -DMAJORVER=${MAJORVER} \
      -DLIB_INSTALL_DIR=%{_libdir} -DINCLUDE_INSTALL_DIR=%{_includedir} \
      -DLIB_PATH=%{_lib} contrib/TFLiteSharp/TFLiteNative
%endif

make %{?_smp_mflags}

cd contrib/TFLiteSharp/
for ASM in %{Assemblies}; do
%dotnet_build $ASM
%dotnet_pack $ASM
done

%install
%make_install
cd contrib/TFLiteSharp/TFLiteSharp
for ASM in %{Assemblies}; do
%dotnet_install $ASM
done

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%manifest %{name}.manifest
%license LICENSE

%files TFLiteNative
%manifest tflite-native.manifest
%{_libdir}/libtflite-native.so*

%files TFLiteNative-devel
%{_includedir}/*
%{_libdir}/pkgconfig/tflite-native.pc
%{_libdir}/libtflite-native.so*

%files TFLiteSharp
%attr(644,root,root) %{dotnet_assembly_files}
