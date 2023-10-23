#!/usr/bin/env bash
#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

# Setup Android Cross-Build Environment (ANDROID SDK)
SCRIPT_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ) # absolute path to directory where script is
INSTALL_PATH=$SCRIPT_HOME/android_sdk # path to directory where android sdk will be installed
PLATFORMS_PACKAGE_VERSION="29" # version of platfroms package which will be installed
BUILD_TOOLS_PACKAGE_VERSION="29.0.3" # version of build-tools package which will be installed
NDK_VERSION="20.0.5594570" # version of ndk which will be installed
COMMAND_LINE_TOOLS_ARCHIVE="commandlinetools-linux-6514223_latest.zip" # command line tools archive name from site https://developer.android.com/studio/#downloads for bootstrap
COMMAND_LINE_TOOLS_VERSION="10.0" # version of command line tools which will be installed


usage() {
 printf "usage: ./build_android_sdk.sh [--option [option_value]]\n"
 printf "  --install-dir                 - absolute path to directory where android sdk will be installed, by default: $INSTALL_PATH\n"
 printf "  --platforms-package-version   - version of platforms package which will be installed, by default: $PLATFORMS_PACKAGE_VERSION\n"
 printf "  --build-tools-package-version - version of build-tools package which will be installed, by default: $BUILD_TOOLS_PACKAGE_VERSION\n"
 printf "  --command-line-tools-version  - version of cmdline-tools package which will be installed, by default: $COMMAND_LINE_TOOLS_VERSION\n"
 printf "  --command-line-tools-archive  - name of command line tools archive from site https://developer.android.com/studio/#downloads, by default: $COMMAND_LINE_TOOLS_ARCHIVE\n"
 printf "  --help                        - show this text\n"
}

check_that_available() {
  local util_name=${1}
  local possible_util_alias=${2}
  if ! [ -x "$(command -v $util_name)" ]; then
    printf "ERROR: this script uses $util_name utility, \n"
    printf "please install it and repeat try (e.g. for ubuntu execute command: sudo apt install $possible_util_alias)\n"
    exit 1
  fi
}

check_preconditions() {
  check_that_available wget wget
  check_that_available unzip unzip
  check_that_available java 'default-jdk or openjdk-{version}-jdk. SDK command line tools require proper openjdk version'
}

check_that_android_sdk_have_not_been_installed_yet() {
  local root=${1}

  if [ -d $root ]; then
    echo "Directory '$root', where android sdk should be installed, exists. Please remove it or define another path"
    exit 1
  fi
}

make_environment() {
  local root=${1}
  check_that_android_sdk_have_not_been_installed_yet $root
  mkdir -p $root

  pushd $root > /dev/null
  export ANDROID_HOME=$root
  export PATH=$PATH:$ANDROID_HOME/cmdline-tools/${COMMAND_LINE_TOOLS_VERSION}/bin
  export PATH=$PATH:$ANDROID_HOME/platform-tools
  export JAVA_OPTS='-XX:+IgnoreUnrecognizedVMOptions'
  popd > /dev/null
}

install_command_line_tools() {
  local root=${1}
  local download_url=https://dl.google.com/android/repository
  temp_dir=$(mktemp -d)
  pushd $temp_dir > /dev/null
  wget $download_url/$COMMAND_LINE_TOOLS_ARCHIVE
  if [ ${?} -ne 0 ]; then
    echo "seems like '$COMMAND_LINE_TOOLS_ARCHIVE' not found. Please, go to https://developer.android.com/studio/#downloads "
    echo "and check name and version of command line tools archive for linux in the table 'Command line tools only' "
    echo "and put it as --command-line-tools-archive-name parameter value"
    exit 1
  fi

  unzip $COMMAND_LINE_TOOLS_ARCHIVE
  rm $COMMAND_LINE_TOOLS_ARCHIVE

  yes | tools/bin/sdkmanager --sdk_root=${root} --licenses
  tools/bin/sdkmanager --sdk_root=${root} "cmdline-tools;${COMMAND_LINE_TOOLS_VERSION}"
  popd > /dev/null

  rm -rf $temp_dir
}

check_that_given_version_of_package_available() {
  # this function assumes that current working directory is directory where android sdk will be installed
  local package_base_name=${1}
  local package_version=${2}

  sdkmanager --sdk_root=${ANDROID_HOME} --list | grep ${package_base_name}${package_version} > /dev/null
  if [ ${?} -ne 0 ]; then
    echo "package '${package_base_name}${package_version}' is not available"
    exit 1
  fi
}

install_android_sdk() {
  local root=${1}

  pushd $root > /dev/null
  check_that_given_version_of_package_available "platforms;android-" ${PLATFORMS_PACKAGE_VERSION}
  check_that_given_version_of_package_available "build-tools;" ${BUILD_TOOLS_PACKAGE_VERSION}
  sdkmanager --sdk_root=${root} "platform-tools" "emulator"
  sdkmanager --sdk_root=${root} "platforms;android-$PLATFORMS_PACKAGE_VERSION"
  sdkmanager --sdk_root=${root} "build-tools;$BUILD_TOOLS_PACKAGE_VERSION"
  popd > /dev/null
}

install_android_ndk() {
  local root=${1}

  pushd $root > /dev/null
  sdkmanager --sdk_root=${root} "ndk;$NDK_VERSION"
  echo "Android NDK is installed on $root"/ndk/$NDK_VERSION" directory"
  popd > /dev/null
}

while [[ $# -gt 0 ]]; do
  key="$(echo $1 | awk '{print tolower($0)}')"
  case "$key" in
    --help)
      usage
      exit 0
      ;;
    --install-dir)
      shift
      INSTALL_PATH=${1}
      shift
      ;;
    --platforms-package-version)
      shift
      PLATFORMS_PACKAGE_VERSION=${1}
      shift
      ;;
    --build-tools-package-version)
      shift
      BUILD_TOOLS_PACKAGE_VERSION=${1}
      shift
      ;;
    --ndk-version)
      shift
      NDK_VERSION=${1}
      shift
      ;;
    --command-line-tools-archive)
      shift
      COMMAND_LINE_TOOLS_ARCHIVE=${1}
      shift
      ;;
    *)
      echo "Invalid option '$1'"
      usage
      exit 1
      ;;
  esac
done

check_preconditions
make_environment $INSTALL_PATH
install_command_line_tools $INSTALL_PATH
install_android_ndk $INSTALL_PATH
install_android_sdk $INSTALL_PATH
