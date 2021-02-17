#include "InputInitializer.h"
#include "TensorView.h"

#include <misc/tensor/IndexIterator.h>

#include <iostream>
#include <fstream>

namespace nnfw
{
namespace onert_cmp
{

void RandomInputInitializer::run(IOManager &manager)
{
  for (uint32_t index = 0; index < manager.inputs(); index++)
  {
    auto info = manager.inputTensorInfo(index);

    switch (info.dtype)
    {
      case NNFW_TYPE_TENSOR_BOOL:
        setValue<bool>(manager, index);
        break;
      case NNFW_TYPE_TENSOR_UINT8:
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
        setValue<uint8_t>(manager, index);
        break;
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
        setValue<int8_t>(manager, index);
        break;
      case NNFW_TYPE_TENSOR_FLOAT32:
        setValue<float>(manager, index);
        break;
      case NNFW_TYPE_TENSOR_INT32:
        setValue<int32_t>(manager, index);
        break;
      case NNFW_TYPE_TENSOR_INT64:
        setValue<int64_t>(manager, index);
        break;
      default:
        std::cerr << "[ ERROR ] "
                  << "Unspported input data type" << std::endl;
        exit(-1);
        break;
    }
  }
}

template <typename T> void RandomInputInitializer::setValue(IOManager &manager, uint32_t tensor_idx)
{
  TensorView<T> tensor_view = manager.inputView<T>(tensor_idx);

  nnfw::misc::tensor::iterate(tensor_view.shape())
    << [&](const nnfw::misc::tensor::Index &ind) { tensor_view.at(ind) = _randgen.generate<T>(); };
}

const int FILE_ERROR = 2;

void FileInputInitializer::run(IOManager &manager)
{
  uint32_t num_inputs = manager.inputs();
  bool read_data = (_files.size() == num_inputs);
  if (!read_data)
  {
    std::cerr << "[ ERROR ] "
              << "Wrong number of input files." << std::endl;
    exit(1);
  }

  for (uint32_t index = 0; index < num_inputs; index++)
  {
    auto path = _files[index];
    auto dest = manager.inputBase(index);

    std::ifstream in(path);
    if (!in.good())
    {
      std::cerr << "can not open data file " << path << "\n";
      exit(FILE_ERROR);
    }
    in.seekg(0, std::ifstream::end);
    size_t len = in.tellg();
    in.seekg(0, std::ifstream::beg);

    assert(dest.size() == len);
    in.read(reinterpret_cast<char *>(dest.data()), len);
  }
}

} // namespace onert_cmp
} // namespace nnfw
