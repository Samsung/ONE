# dio-hdf5

_dio-hdf5_ is a library to help loading hdf5 files (_dio_ indicates data I/O).

The hdf5 file should have the following structure.

```
Group "/"
 > Group <group_name>
   > Group <data_idx>
     > Dataset <input_idx>
```

## Example

```cpp
dio_hdf5::HDF5Importer h5{input_path};

h5.importGroup("value");

// Prepare buffer
const uint32_t input_byte_size = 16;
std::vector<char> buffer(input_byte_size);

// Write the first input of the first data to buffer
readTensor(0, 0, buffer.data());

DO_SOMETHING_WITH(buffer);
```
