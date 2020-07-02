#!/bin/bash

set -u

command_exists() {
	command -v "$@" > /dev/null 2>&1
}

progname=$(basename "${BASH_SOURCE[0]}")
indir="."
outdir="."
nnpkg_run=${nnpkg_run:-"Product/out/bin/nnpackage_run"}
difftool=${difftool:-"h5diff"}
delete_dumped_on_failure=0

usage() {
  echo "Usage: $progname [options] nnpackage_test"
  echo "Run an nnpackage testcase"
  echo ""
  echo "Returns"
  echo "     0       success"
  echo "  non-zero   failure"
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -i   set input directory (default=$indir)"
  echo "    -o   set output directory (default=$outdir)"
  echo "    -d   delete dumped file on failure."
  echo "         (dumped file are always deleted on success) (default=$delete_dumped_on_failure)"
  echo ""
  echo "Environment variables:"
  echo "   nnpackage_run    path to nnpackage_run (default=Product/out/bin/nnpackage_run)"
  echo "   difftool         path to i5diff or h5diff (default=h5diff)"
  echo ""
  echo "Examples:"
  echo "    $progname Add_000                => run $indir/Add_000 and check output"
  echo "    $progname -i nnpkg-tcs Add_000   => run nnpkg-tcs/Add_000 and check output"
  exit 1
}

if [ $# -eq 0 ]; then
  echo "For help, type $progname -h"
  exit 1
fi

while getopts "hdi:o:" OPTION; do
case "${OPTION}" in
    h) usage;;
    d) delete_dumped_on_failure=1;;
    i) indir=$OPTARG;;
    o) outdir=$OPTARG;;
    ?) exit 1;;
esac
done

shift $((OPTIND-1))

if [ $# -ne 1 ]; then
  echo "error: wrong argument (no argument or too many arguments)."
  echo "For help, type $progname -h"
  exit 1
fi

if [ ! -e Product ]; then
  echo "error: please make sure to run this script in nnfw home."
  exit 1
fi

tcname=$(basename "$1")
nnpkg="$indir/$tcname"

# run

if [ ! -e $nnpkg ]; then
  echo "error: nnpackage "$nnpkg" does not exist."
  exit 1
fi

if ! command_exists $nnpkg_run; then
  echo "error: runner "$nnpkg_run" does not exist."
  exit 1
fi

dumped="$outdir/$tcname".out.h5

echo -n "[  Run  ] $nnpkg "

if $nnpkg_run \
--nnpackage "$nnpkg" \
--load "$nnpkg/metadata/tc/input.h5" \
--dump "$dumped" >& /dev/null > "$dumped.log" 2>&1 ; then
  echo -e "\tPass"
  rm "$dumped.log"
else
  echo -e "\tFail"
  echo ""
  cat "$dumped.log"
  echo ""
  rm "$dumped.log"
  exit 2
fi

# diff

if ! command_exists $difftool; then
  echo "error: difftool "$difftool" does not exist."
  exit 1
fi

expected="$nnpkg/metadata/tc/expected.h5"

echo -n "[Compare] $nnpkg "

test_fail()
{
  echo -e "\tFail"
  [ $delete_dumped_on_failure ] && rm "$dumped"
  cat "$dumped.log"
  rm "$dumped.log"
  exit 3
}

test_pass()
{
  echo -e "\tPass"
  cat "$dumped.log"
  rm "$dumped" "$dumped.log"
}

if ! $difftool -d 0.001 -v "$dumped" "$expected" /value >& "$dumped.log"; then
  test_fail
elif grep "not comparable" "$dumped.log" > /dev/null; then
  test_fail
else
  test_pass
fi
