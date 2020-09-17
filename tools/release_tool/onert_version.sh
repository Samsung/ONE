#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nnfw_root="$( cd "${script_dir%*/*/*}" && pwd )"
nightly=1

usage() {
  echo "Usage: $progname version"
  echo "Update or show onert version information"
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -n   show current onert version with nightly suffix"
  echo "    -s   set onert version"
  echo ""
  echo "Examples:"
  echo "    $progname           => show current onert version"
  echo "    $progname -s 1.6.0  => set onert version info in all sources"
  exit 1
}

show_version() {
  version_line=$(cat ${nnfw_root}/packaging/nnfw.spec | grep "Version:")
  current_version=${version_line#"Version:"}

  if [ $nightly -eq 0 ]; then
    # Get head commit's date
    pushd $nnfw_root > /dev/null
    date=$(git log -1 --format=%ad --date=format:%y%m%d)
    echo $current_version-nightly-$date
    popd > /dev/null
  else
    echo $current_version
  fi

  exit 0
}

set_version() {
  version=$1
  perl -pi -e "s/^release = .*/release = \'$version\'/" ${nnfw_root}/docs/conf.py
  perl -pi -e "s/^Version: .*/Version: $version/" ${nnfw_root}/packaging/nnfw.spec

  IFS=. read M m p <<< "$version"
  hex=$(printf '0x%08x' $(( (($M << 24)) | (($m << 8)) | $p )))
  perl -pi -e "s/^#define NNFW_VERSION.*/#define NNFW_VERSION $hex/" ${nnfw_root}/runtime/onert/api/include/nnfw_version.h

  perl -pi -e "s/versionName .*$/versionName \"$version\"/" ${nnfw_root}/runtime/contrib/android/api/build.gradle
}

if [ $# -eq 0 ]; then
  show_version
fi

while getopts "hns:" OPTION; do
case "${OPTION}" in
    h) usage;;
    n) nightly=0; show_version;;
    s) set_version "$OPTARG";;
    ?) exit 1;;
esac
done

shift $((OPTIND-1))
