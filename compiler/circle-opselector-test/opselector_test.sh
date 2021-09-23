#!/bin/bash
# compiler/circle-opselector/test/opselector_test.sh while_dynamic.tflite "1 2 3 4"
# compiler/circle-opselector/test/opselector_test.sh tflite_files/Part_Sqrt_Rsqrt_001.tflite "0 1 2"

red=`tput setaf 1`
blue=`tput setaf 4`
VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $VERIFY_SOURCE_PATH/../../
ROOT_PATH=`pwd`

TFLITE_NAME="$1"
NODES=$2
TFLITE_PATH=$ROOT_PATH/build/compiler/common-artifacts/$TFLITE_NAME
function error_case(){
	value=$1
  	echo "${red}There are some problem : $value"
  	rm -rf $TEMP_PATH
  	rm -rf $ROOT_PATH/trimmed
  	exit 1
}

mkdir "$ROOT_PATH/test_tmp"
TEMP_PATH="$ROOT_PATH/test_tmp"

# trim tflite
echo $NODES > $TEMP_PATH/opcodelist.txt
if ! python3 $ROOT_PATH/tools/tflitefile_tool/select_operator.py $TFLITE_PATH $TEMP_PATH/opcodelist.txt $TEMP_PATH/trimmed.tflite  > /dev/null; then
	error_case "can't select tflite file(select_operator.py)"
fi

# tflite to circle
$ROOT_PATH/build/compiler/tflite2circle/tflite2circle $TFLITE_PATH $TEMP_PATH/origin.circle

# trim circle
if ! eval $ROOT_PATH/build/compiler/circle-opselector/opselector --by_id \"$NODES\" --input $TEMP_PATH/origin.circle --output $TEMP_PATH/trimmed.circle  > /dev/null; then
	error_case "can't select circle file(circle-opselector)"
fi

# tflite to nnpacakge and create golden value
if ! $ROOT_PATH/tools/nnpackage_tool/sth2nnpkgtc/tflite2nnpkgtc.sh -o $TEMP_PATH $TEMP_PATH/trimmed.tflite  > /dev/null; then
	error_case "can't convert tflite to nnpackage"
fi

if ! [ -e "$TEMP_PATH/trimmed/metadata/tc/input.h5" ]; then
	error_case "can't create golden value"
fi

# convert tflite in nnpackage to circle 
sed -i 's/tflite/circle/g' $TEMP_PATH/trimmed/metadata/MANIFEST
rm $TEMP_PATH/trimmed/trimmed.tflite
mv $TEMP_PATH/trimmed.circle $TEMP_PATH/trimmed/

# nnpackage to root folder(because of onert-test)
mv $TEMP_PATH/trimmed $ROOT_PATH

# compare trimmed tflite and trimmed circle files
if ! $ROOT_PATH/Product/out/test/onert-test nnpkg-test trimmed > /dev/null; then
	echo "${red}test fail -- $TFLITE_NAME, $NODES"
	rm -r $TEMP_PATH
	rm -r $ROOT_PATH/trimmed
	exit 1
else
	echo "${blue}test success -- $TFLITE_NAME, $NODES"
fi
rm -r $TEMP_PATH
rm -r $ROOT_PATH/trimmed
