#!/bin/bash
# This script is to automate the process of monthly release with github API

# Test if getopt is enhanced version
getopt --test > /dev/null
if [ $? -ne 4 ]; then
  echo "[ERROR] Your system doesn't have enhanced getopt"
  exit 2
fi

function Usage()
{
  echo "Usage: ./$(basename ${BASH_SOURCE[0]}) --tag TAG --release_note RELEASE_NOTE \
--token TOKEN [--release_name RELEASE_NAME] [--commitish COMMITISH] [--draft] \
[--host_name HOST_NAME] [--repo_owner REPO_OWNER] [--repo_name REPO_NAME] [--asset] ..."
  echo ""
  echo "[OPTIONS]"
  echo "--tag              The name of the tag"
  echo "--release_name     The name of the release"
  echo "--release_note     Path of text file describing the contents of the release"
  echo "--commitish        The commitish value that determines where the Git tag is created from"
  echo "--draft            Create a draft release"
  echo "--token            User token for authentication"
  echo "--host_name        Host name for endpoint URL [Enterprise-specific endpoint only]"
  echo "--repo_owner       Owner of the repository"
  echo "--repo_name        The name of the repository"
  echo "--asset            Path of release asset"
  echo "--asset_url        URL from which release asset is downloaded"
  echo ""
  echo "[EXAMPLE]"
  echo "$ ./git_release.sh --tag 1.9.0 --commitish release/1.9.0 --token 0de25f1ca5d1d758fe877b18c06 \\"
  echo "  --repo_owner mhs4670go --repo_name test_repo --release_note local/repo/release_note \\"
  echo "  --asset ONE-compiler.tar.gz --asset ONE-runtime.tar.gz"
  echo ""
  echo "$ ./git_release.sh --tag v1.1 --commitish c024e85d0ce6cb1ed2fbc66f1a9c1c2814da7575 \\"
  echo "  --token 0de25f1ca5d1d758fe877b18c06 --repo_owner Samsung --repo_name ONE \\"
  echo "  --release_name \"Release Automation\" --release_note /home/mhs4670go/ONE/release_doc \\"
  echo "  --host_name github.sec.company.net --draft \\"
  echo "  --asset_url \"http://one.server.com/artifacts/ONE-compiler.tar.gz\""
  echo ""
  echo "[REFERENCE]"
  echo "https://developer.github.com/v3/repos/releases/#create-a-release"

}

SHORT_OPTS=h
LONG_OPTS="\
help,\
tag:,\
release_name:,\
release_note:,\
commitish:,\
draft,\
token:,\
host_name:,\
repo_owner:,\
repo_name:,\
asset:,\
asset_url:"

OPTS=$(getopt --options "$SHORT_OPTS" --longoptions "$LONG_OPTS" --name "$0" -- "$@")

if [ $? != 0 ] ; then echo "[ERROR] Failed to parse options" ; exit 2 ; fi

eval set -- "$OPTS"

unset TAG_NAME
unset RELEASE_NAME
unset RELEASE_NOTE
unset TARGET_COMMITISH
unset USER_TOKEN
unset HOST_NAME
unset REPO_OWNER
unset REPO_NAME
IS_DRAFT=false
ASSET_PATHS=()
ASSET_URLS=()

while true ; do
  case "$1" in
    -h|--help )
      Usage
      exit 0
      ;;
    --tag ) # REQUIRED
      TAG_NAME="$2"
      shift 2
      ;;
    --release_name )
      RELEASE_NAME="$2"
      shift 2
      ;;
    --release_note ) # REQUIRED
      RELEASE_NOTE="$2"
      shift 2
      ;;
    --commitish )
      TARGET_COMMITISH="$2"
      shift 2
      ;;
    --draft )
      IS_DRAFT=true
      shift
      ;;
    --token ) # REQUIRED
      USER_TOKEN="$2"
      shift 2
      ;;
    --host_name )
      HOST_NAME="$2/api/v3"
      shift 2
      ;;
    --repo_owner )
      REPO_OWNER="$2"
      shift 2
      ;;
    --repo_name )
      REPO_NAME="$2"
      shift 2
      ;;
    --asset )
      ASSET_PATHS+=("$2")
      shift 2
      ;;
    --asset_url )
      ASSET_URLS+=("$2")
      shift 2
      ;;
    -- )
      shift
      break
      ;;
    *)
      echo "[ERROR] getopt internal error"
      exit 2
      ;;
  esac
done

# Check if required options are specified
if [ -z ${TAG_NAME} ]; then
  echo "[ERROR] You must specify '--tag' option"
  Usage
  exit 0
fi
if [ -z ${RELEASE_NOTE} ]; then
  echo "[ERROR] You must specify '--release_note' option"
  Usage
  exit 0
fi
if [ -z ${USER_TOKEN} ]; then
  echo "[ERROR] You must specify '--token' option"
  Usage
  exit 0
fi

ASSETS_FROM_URL=()
# Get asset name from url
for ASSET_URL in "${ASSET_URLS[@]}"; do
  ASSETS_FROM_URL+=($(basename "${ASSET_URL}"))
done

# Print variables and set default value
DEFAULT_RELEASE_NAME="ONE Release ${TAG_NAME}"
DEFAULT_HOST_NAME="api.github.com"
DEFAULT_REPO_OWNER="Samsung"
DEFAULT_REPO_NAME="ONE"
echo "======================[RELEASE INFO]======================"
echo "TAG_NAME         : ${TAG_NAME}"
echo "RELEASE_NAME     : ${RELEASE_NAME:=${DEFAULT_RELEASE_NAME}}"
echo "RELEASE_NOTE     : ${RELEASE_NOTE}"
echo "TARGET_COMMITISH : ${TARGET_COMMITISH:=${TAG_NAME}}"
echo "IS_DRAFT         : ${IS_DRAFT}"
echo "USER_TOKEN       : ${USER_TOKEN}"
echo "HOST_NAME        : ${HOST_NAME:=${DEFAULT_HOST_NAME}}"
echo "REPO_OWNER       : ${REPO_OWNER:=${DEFAULT_REPO_OWNER}}"
echo "REPO_NAME        : ${REPO_NAME:=${DEFAULT_REPO_NAME}}"
echo "ASSETS           : ${ASSET_PATHS[@]}"
echo "ASSETS_FROM_URL  : ${ASSETS_FROM_URL[@]}"
echo "==========================================================="

function generate_release_data()
{
  cat <<EOF
{
  "tag_name": "${TAG_NAME}",
  "target_commitish": "${TARGET_COMMITISH}",
  "name": "${RELEASE_NAME}",
  "body": "$(cat $1 | sed 's/$/\\n/' | tr -d '\n')",
  "draft": ${IS_DRAFT},
  "prerelease": false
}
EOF
}

# Check if the release already exists
RELEASE_URL=$(curl -s --request GET --header "Authorization: token ${USER_TOKEN}" \
https://${HOST_NAME}/repos/${REPO_OWNER}/${REPO_NAME}/releases/tags/${TAG_NAME} | \
jq -r '.url')

if [ "$RELEASE_URL" != null ]; then
  echo "[ERROR] The tag name you specified already exists."
  exit 2
fi

# Create a release (with assinging upload_url using jq)
UPLOAD_URL=$(curl -s --request POST --header "Authorization: token ${USER_TOKEN}" \
--header "Accept: application/json" \
--data "$(eval generate_release_data '${RELEASE_NOTE}')" \
"https://${HOST_NAME}/repos/${REPO_OWNER}/${REPO_NAME}/releases" | \
jq -r '.upload_url')

UPLOAD_URL=$(echo ${UPLOAD_URL} | cut -d "{" -f 1)?name=

# Download assets from url
TMPDIR=$(mktemp -d)
pushd $TMPDIR
for ASSET_URL in "${ASSET_URLS[@]}"; do
  wget "$ASSET_URL"
done
popd

# Upload the assets from url
for ASSET_NAME in "${ASSETS_FROM_URL[@]}"; do
  ASSET_PATH="${TMPDIR}/${ASSET_NAME}"
  curl -s --request POST --header "Authorization: token ${USER_TOKEN}" \
  --header "Content-Type: $(file -b --mime-type ${ASSET_PATH})" \
  --data-binary @${ASSET_PATH} \
  ${UPLOAD_URL}${ASSET_NAME} > /dev/null
done

rm -rf ${TMPDIR}

# Upload the assets from local
for ASSET_PATH in "${ASSET_PATHS[@]}"; do
  ASSET_BASENAME=$(basename ${ASSET_PATH})
  curl -s --request POST --header "Authorization: token ${USER_TOKEN}" \
  --header "Content-Type: $(file -b --mime-type ${ASSET_PATH})" \
  --data-binary @${ASSET_PATH} \
  ${UPLOAD_URL}${ASSET_BASENAME} > /dev/null
done
