# cwrap

_cwrap_ is a collection of C++ wrappers for POSIX C API.

## How to use

Currently it supports only file descriptor.

## Example
- File Descriptor

```cpp
cwrap::Fildes fildes{open(path.c_str(), O_RDONLY)};

if (fildes.get() < 0)
{
    std::ostringstream ostr;
    ostr << "Error: " << path << " not found" << std::endl;
    throw std::runtime_error{ostr.str()};
}

google::protobuf::io::FileInputStream fis(fildes.get());
```
