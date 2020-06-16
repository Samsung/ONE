#!/usr/bin/env bash

# TODO Reuse the fuction in run_test.sh. This is its duplication.
function need_download()
{
    LOCAL_PATH=$1
    REMOTE_URL=$2
    if [ ! -e $LOCAL_PATH ]; then
        return 0;
    fi
    # Ignore checking md5 in cache
    if [ ! -z $IGNORE_MD5 ] && [ "$IGNORE_MD5" == "1" ]; then
        return 1
    fi

    LOCAL_HASH=$(md5sum $LOCAL_PATH | awk '{ print $1 }')
    REMOTE_HASH=$(curl -ss $REMOTE_URL | md5sum  | awk '{ print $1 }')
    # TODO Emit an error when Content-MD5 field was not found. (Server configuration issue)
    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "Downloaded file is outdated or incomplete."
        return 0
    fi
    return 1
}

# TODO Reuse the fuction in run_test.sh. This is its duplication.
download_tests()
{
    SELECTED_TESTS=$@

    echo ""
    echo "Downloading tests:"
    echo "======================"
    for TEST_NAME in $SELECTED_TESTS; do
        echo $TEST_NAME
    done
    echo "======================"

    for TEST_NAME in $SELECTED_TESTS; do
        # Test configure initialization
        MODELFILE_SERVER_PATH=""
        MODELFILE_NAME=""
        source $TEST_ROOT_PATH/$TEST_NAME/config.sh

        TEST_CACHE_PATH=$CACHE_ROOT_PATH/$TEST_NAME
        MODELFILE=$TEST_CACHE_PATH/$MODELFILE_NAME
        MODELFILE_URL="$MODELFILE_SERVER/$MODELFILE_NAME"
        if [ -n  "$FIXED_MODELFILE_SERVER" ]; then
            MODELFILE_URL="$FIXED_MODELFILE_SERVER/$MODELFILE_NAME"
        fi

        # Download model file
        if [ ! -e $TEST_CACHE_PATH ]; then
            mkdir -p $TEST_CACHE_PATH
        fi

        # Download unless we have it in cache (Also check md5sum)
        if need_download "$MODELFILE" "$MODELFILE_URL"; then
            echo ""
            echo "Download test file for $TEST_NAME"
            echo "======================"

            rm -f $MODELFILE # Remove invalid file if exists
            pushd $TEST_CACHE_PATH
            wget -nv $MODELFILE_URL
            if [ "${MODELFILE_NAME##*.}" == "zip" ]; then
                unzip -o $MODELFILE_NAME
                rm *.zip
            fi
            popd
        fi

    done
}

realpath()
{
  readlink -e -- "$@"
}

usage()
{
    echo "Usage: $0 --modelfile-server=MODELFILE_SERVER --install-path=INSTALL_DIR"
    echo "  MODELFILE_SERVER : Base URL of the model file server"
    echo "  INSTALL_DIR      : Path to be installed"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$(echo $1 | awk '{print tolower($0)}')"
    case "$key" in
        -?|-h|--help)
            usage
            exit 1
            ;;
        --modelfile-server)
            MODELFILE_SERVER="$2"
            shift
            ;;
        --modelfile-server=*)
            MODELFILE_SERVER="${1#*=}"
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            ;;
        *)
            echo "Invalid option '$1'"
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -z "$MODELFILE_SERVER" ]; then
    echo "Please specify a value for --modelfile-server or MODELFILE_SERVER(env)."
    usage
    exit 1
fi

if [ -z "$INSTALL_DIR" ]; then
    echo "Please specify a value for --install-dir or INSTALL_DIR(env)."
    usage
    exit 1
fi

set -e

THIS_SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE}))
source ${THIS_SCRIPT_DIR}/../common.sh

CACHE_ROOT_PATH=$INSTALL_DIR
FIXED_MODELFILE_SERVER="${MODELFILE_SERVER:-}"
TEST_ROOT_PATH=${THIS_SCRIPT_DIR}/models

# All models in the directory are the target models
pushd ${TEST_ROOT_PATH}
MODELS=$(ls -d */)
popd

download_tests $MODELS

set +e
