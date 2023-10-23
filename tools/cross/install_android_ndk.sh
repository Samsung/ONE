#!/usr/bin/env bash
#
# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Setup Android Cross-Build Environment

usage()
{
    echo "Usage: $0 [--ndk-version=NDKVersion] [--install-dir=InstallDir]"
    echo "  NDKVersion : r20(default) or higher"
    echo "  InstallDir : Path to be installed"
    exit 1
}

__CrossDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
__NDKVersion=r20
__UnprocessedBuildArgs=

while [[ $# -gt 0 ]]
do
    key="$(echo $1 | awk '{print tolower($0)}')"
    case "$key" in
        -?|-h|--help)
            usage
            exit 1
            ;;
        --ndk-version)
            __NDKVersion="$2"
            shift
            ;;
        --ndk-version=*)
            __NDKVersion="${1#*=}"
            ;;
        --install-dir)
            __InstallDir="$2"
            shift
            ;;
        --install-dir=*)
            __InstallDir="${1#*=}"
            ;;
        *)
            echo "Invalid option '$1'"
            usage
            exit 1
            ;;
    esac
    shift
done

__InstallDir=${__InstallDir:-"${__CrossDir}/ndk"}

NDK_DIR=android-ndk-${__NDKVersion}
NDK_ZIP=${NDK_DIR}-linux-x86_64.zip

if [[ -e $__InstallDir ]]; then
  echo "ERROR: $__InstallDir already exists"
  exit 1
fi

echo "Downloading Android NDK ${__NDKVersion}"
mkdir -p "$__InstallDir"

wget -nv -nc https://dl.google.com/android/repository/$NDK_ZIP -O $__InstallDir/$NDK_ZIP

if [ $? -ne 0 ]; then
    echo "Failed downloading. Please check NDK version and network connection."
    exit 1
fi

echo "Unzipping Android NDK"
unzip -qq -o $__InstallDir/$NDK_ZIP -d $__InstallDir
rm $__InstallDir/$NDK_ZIP
