# Content

- git_release.sh
- onert_version.sh

# git_release.sh

This tool helps you to automate GitHub releases.

## Usage
```
$ ./git_release.sh --tag TAG --release_note RELEASE_NOTE \
--token TOKEN [--release_name RELEASE_NAME] [--commitish COMMITISH] [--draft] \
[--host_name HOST_NAME] [--repo_owner REPO_OWNER] [--repo_name REPO_NAME] [--asset] ...
```

## Options
```
--tag              The name of the tag
--release_name     The name of the release
--release_note     Path of text file describing the contents of the release
--commitish        The commitish value that determines where the Git tag is created from
--draft            Create a draft release
--token            User token for authentication
--host_name        Host name for endpoint URL [Enterprise-specific endpoint only]
--repo_owner       Owner of the repository
--repo_name        The name of the repository
--asset            Path of release asset
```

## Examples
```
$ ./git_release.sh --tag 1.9.0 --commitish release/1.9.0 --token 0de25f1ca5d1d758fe877b18c06 \
  --repo_owner mhs4670go --repo_name test_repo --release_note local/repo/release_note \
  --asset ONE-compiler.tar.gz --asset ONE-runtime.tar.gz"

$ ./git_release.sh --tag v1.1 --commitish c024e85d0ce6cb1ed2fbc66f1a9c1c2814da7575 \
  --token 0de25f1ca5d1d758fe877b18c06 --repo_owner Samsung --repo_name ONE \
  --release_name "Release Automation" --release_note /home/mhs4670go/ONE/release_doc \
  --host_name github.sec.company.net --draft
```

## Reference
https://developer.github.com/v3/repos/releases/#create-a-release


# onert_version.sh

onert_version.sh updates version information.

## Usage
```
$ ./onert_version.sh -h
Usage: onert_version.sh version
Update or show onert version information
```

## Options
```
-h   show this help
-s   set onert version
```

## Examples
```
$ ./onert_version.sh           => show current onert version
$ ./onert_version.sh -s 1.6.0  => set onert version info in all sources
```
