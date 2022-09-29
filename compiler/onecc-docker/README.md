# onecc-docker

_onecc-docker_ broadens ONE tools to be used in other platforms.

## Description

For now, ONE tools only support Ubuntu 18.04 and 20.04(not officially).
So, it is difficult for people in different environments to use our tools without using ubuntu 18.04.

To overcome this limitation, we provide _onecc-docker_ that runs using a Docker so that users can use ONE tools more widely.

This tool aims at the following objectives.

- Unsupported Ubuntu OS supports ONE tools
- Install and use ONE tools lightly and quickly using Docker

## Requirements

- Any Linux distribution
- Docker
    - Requires root privileges.
           - _onecc-docker_ requires the current `user ID` to be included in the `Docker group` because it requires the Docker-related commands to be executed without `sudo` privileges.
             - See "[Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)"
- Python 3.8
  - requests

## Note

_onecc-docker_ is currently in incubation stage.

The onecc-docker debian package should be created with one-compiler debian package when ONE
compiler project builds. To this end, it is correct to configure the onecc-docker debian codes in
./infra/debian/compiler directory. However, we are currently working on the code, so we will
temporarily implement it in this location.

TODO: Merge this debian directory into ./infra/debian/compiler code.
