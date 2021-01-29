### Prerequired jenkins server setting

#### Node label

- `checker`
  - Usage: format checker & commit checker
  - Requirement: `docker` package and permission for jenkins
- `builder`
  - Usage: Build server
  - Requirement: `docker` package and permission for jenkins
- `x64_test_slave`
  - Usage: Test on x86_64
  - Requirement: `docker` package and permission for jenkins
- `xu4_test_slave_1804`
  - Usage: Test on armv7l ubuntu 18.04
  - Requirement: `curl`, `wget`, `libboost-all-dev`, `libhdf5-dev`, `hdf5-tools` package on ubuntu 18.04
- `n2-test-slave`
  - Usage: Test on aarch64 ubuntu 18.04
  - Requirement: `curl`, `wget`, `libboost-all-dev`, `libhdf5-dev`, `hdf5-tools` package on ubuntu 18.04

#### Docker registry credentials

Set docker credential in project folder's configure

#### Credentials: secret text
