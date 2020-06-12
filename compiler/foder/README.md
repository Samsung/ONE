# foder

_foder_ is a header only library that loads files.

## Example

```cpp
foder::FileLoader fileloader{input_path};

std::vector<char> data = fileloader.load();

const circle::Model *circle_model = circle::GetModel(data.data());
```
