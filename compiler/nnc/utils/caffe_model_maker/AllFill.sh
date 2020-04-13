#!/bin/sh
: '
Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'


#Fills all models and writes errors
usage () { 
    echo "Filler.sh should be in the working directory\nusage:
    no args - assumes current directory
    -d=<dir> fills models in <dir>
    Example:
    $(basename $0) -d='./foobar/'"
}

DIR="./"
for i in "$@"
do
    case $i in
        -h|--help|help)
            usage
            exit 1
            ;;
        -d=*)
            DIR=${i#*=}
            ;;
    esac
    shift
done
echo $DIR
if [ $# -eq 0 ]; then
    echo "Assume working directory"
fi
for a in `ls $DIR*.prototxt`; do
  ./Filler.sh $a
done 2>error.log
