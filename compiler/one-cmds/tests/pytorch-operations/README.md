## Overview 

This directory contains auxilliary tests for small pytorch target models.

Most of the models contains single operations, but some contains multiple operations, that represents one operation with complex semantics.

Models for these tests are taken from res/PyTorchExamples.

## To run all tests

Steps:
1) run 'one-prepare-venv' in bin folder to prepare python virtual-env with TensorFlow
   - you need to run this only once
   - read 'doc/how-to-prepare-virtualenv.txt' for more information
   ```
   bin/one-prepare-venv
   ```
2) run 'test/pytorch-operations/prepare_test_materials.sh' to download test material models
   - you need to run this only once
   - you need internet connection to download files
   - you may need to install 'wget' and 'unzip' packages
   ```
   test/pytorch-operations/prepare_test_materials.sh
   ```
3) run 'test/pytorch-operations/runtestall.sh' to run the test
   ```
   test/pytoch-operations/runtestall.sh
   ```
