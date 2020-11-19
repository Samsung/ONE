# Stab - Static Backend Scheduler

`Stab` is a tool to schedule backend for each opration using profiled data

nnpackage with backend configuration will be created at `./tools/stab/nnpkg_sched`

## Scheduling Process

1. Upload ONE runtime and nnpackage to remote device
   - Use `/tmp/ONE` folder on remote device
1. Profile execution time of each backend on remote device
1. Get profile result from remote device
   - Profile result is saved at `./tools/stab/traces` on host
1. Scheduler backend for each operation to get fastest inference time
   - Use fastest backend for each operation
1. Generate nnpackage with backend configuration
   - Generated at `./tools/stab/nnpkg_sched`

## Prerequisite

- Install Python>=3. Tested on Python 3.6.9 and 3.7.5
- Register SSH keys to use ssh commands without entering password
  ```bash
  ssh-keygen -t rsa
  ssh-copy-id -i ~/.ssh/id_rsa.pub remote_user@remote_ip
  ```

## Usage

```
Usage: python3 ./tools/stab/stab.py --nnpackage nnpackage_dir --ip <IP>
Runs nnpackage on remote device and create nnpackaged with scheduled backends

required arguments:
  --nnpackage NNPACKAGE
                        nnpackage folder to profile
  --ip IP               IP address of remote client

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_THREADS, --num_threads NUM_THREADS
                        Number of threads used by one runtime
  -u USER, --user USER  User of remote client
  -v, --verbose         Print verbose message
  --no-profile          Disable profiling

Examples:
    python3 ./tools/stab/stab.py --nnpackage ../nnpkg_tst/inception --ip 1.1.1.1               => Profile on remote device 1.1.1.1 with current user
    python3 ./tools/stab/stab.py --nnpackage ../nnpkg_tst/inception --ip 1.1.1.1 -n 4          => Profile on remote device 1.1.1.1 using 4 thread for ONE runtime
    python3 ./tools/stab/stab.py --nnpackage ../nnpkg_tst/inception --ip 1.1.1.1 --user odroid => Profile on remote device 1.1.1.1 with user odroid
```
