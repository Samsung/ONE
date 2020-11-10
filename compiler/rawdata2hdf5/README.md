# rawdata2hdf5

_rawdata2hdf5_ is a tool to convert raw data (assumed to be pre-processed) to an hdf5 file.

## Prerequisite
- Raw data pre-processed for the corresponding DNN model
- List of data to convert (saved in the text file)
- Python installed with _numpy_ and _h5py_ (See docs/how-to-prepare-virtualenv.txt)

## Example
```
./rawdata2hdf5 \
> --data_list=tmp/data/datalist.txt
> --output_path=tmp/data/data.h5
```

## Arguments
```
  -h, --help            Show this help message and exit
  -l DATA_LIST, --data_list DATA_LIST
                        Path to the text file which lists the absolute paths of the raw data files to be converted.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the output hdf5 file.
```
