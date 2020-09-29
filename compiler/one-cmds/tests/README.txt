one-cmds testing
================

Run 'runtestall.sh' program to test ONE command line programs, all at once.

Steps:
1) run 'one-prepare-venv' in bin folder to prepare python virtual-env with TensorFlow
  - you need to run this only once
  - read 'doc/how-to-prepare-virtualenv.txt' for more information
----------------------------------------------  
cd ../bin
./one-prepare-venv
----------------------------------------------

2) change back to this test folder
----------------------------------------------
cd ../test
----------------------------------------------

3) run 'prepare_test_materials.sh' to download test material models
  - you need to run this only once
  - you need internet connection to download files
  - you may need to install 'wget' and 'unzip' packages
----------------------------------------------
./prepare_test_materials.sh
----------------------------------------------

4) run 'runtestall.sh' to run the test
----------------------------------------------
./runtestall.sh
----------------------------------------------

End.
