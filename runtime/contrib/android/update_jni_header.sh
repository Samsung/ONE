#!/bin/bash

set -x
set -e

./gradlew assembleRelease -x externalNativeBuildRelease

pushd api/build/intermediates/javac/release/classes
javah com.samsung.onert.NativeSessionWrapper
popd

mv -vf api/build/intermediates/javac/release/classes/com_samsung_onert_NativeSessionWrapper.h api/src/main/native/onert-native-api.h
