These scripts can be useful for developing/testing nnc. Usage and purpose of the scripts can be found in comments in their source code.

Note that these scripts are just development artifacts and are not supposed to go into production in any form.

infer_testcases.py: run inference with `nnkit` on testcases
res2bin.py: used by infer_testcases.py to convert resulting hdf5 to binary format

'testcases' folder structure:
At the moment we use the following structure: a folder for a model contains 'models' and 'testcases' subfolders. The 'models' subfolder contains model that we run inference on, 'testcases' subfolder contains a 'testcase*' folder for each different testcase. Each of those folders in turn contain 'input' with a '.JPEG' file (and '.hdf5' and '.dat' files after running `jpeg2hdf5` script), and 'output' folder where inference results are stored.
