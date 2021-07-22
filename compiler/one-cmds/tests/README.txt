one-cmds testing
================

Run 'runtestall.sh' program to test ONE command line programs, all at once.

Steps:
1) run 'one-prepare-venv' in bin folder to prepare python virtual-env with TensorFlow
  - you need to run this only once
  - read 'doc/how-to-prepare-virtualenv.txt' for more information
----------------------------------------------
bin/one-prepare-venv
----------------------------------------------

2) run 'tests/prepare_test_materials.sh' to download test material models
  - you need to run this only once
  - you need internet connection to download files
  - you may need to install 'wget' and 'unzip' packages
----------------------------------------------
tests/prepare_test_materials.sh
----------------------------------------------

3) run 'test/runtestall.sh' to run the test
----------------------------------------------
test/runtestall.sh
----------------------------------------------

End.
