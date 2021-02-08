#include "IOManager.h"
#include "TensorView.h"

#include <iostream>

#define NNFW_ASSERT_FAIL(expr, msg)   \
  if ((expr) != NNFW_STATUS_NO_ERROR) \
  {                                   \
    std::cerr << msg << std::endl;    \
    exit(-1);                         \
  }

namespace nnfw
{
namespace onert_cmp
{

namespace
{

size_t sizeOfNnfwType(NNFW_TYPE type)
{
  switch (type)
  {
    case NNFW_TYPE_TENSOR_BOOL:
    case NNFW_TYPE_TENSOR_UINT8:
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
      return 1;
    case NNFW_TYPE_TENSOR_FLOAT32:
    case NNFW_TYPE_TENSOR_INT32:
      return 4;
    case NNFW_TYPE_TENSOR_INT64:
      return 8;
    default:
      throw std::runtime_error{"Invalid tensor type"};
  }
}

size_t sizeOfTensor(nnfw_tensorinfo &info)
{
  uint32_t size = sizeOfNnfwType(info.dtype);
  for (int32_t axis = 0; axis < info.rank; axis++)
  {
    size *= info.dims[axis];
  }

  return size;
}

} // namespace

void IOManager::getInput(uint32_t index)
{
  auto &info = _input_infos[index];
  NNFW_ASSERT_FAIL(nnfw_input_tensorinfo(_session, index, &info),
                   "[ ERROR ] Failure during get input data info");

  _inputs.emplace(index, sizeOfTensor(info));
}

void IOManager::getOutput(uint32_t index)
{
  auto &info = _output_infos[index];
  NNFW_ASSERT_FAIL(nnfw_output_tensorinfo(_session, index, &info),
                   "[ ERROR ] Failure during get output data info");

  _outputs.emplace(index, sizeOfTensor(info));
}

template <typename T> TensorView<T> IOManager::inputView(uint32_t index)
{
  if (_inputs.find(index) == _inputs.end())
  {
    getInput(index);
  }

  assert(_input_infos.find(index) != _input_infos.end());
  auto &info = _input_infos.at(index);

  nnfw::misc::tensor::Shape shape(info.rank);
  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    shape.dim(axis) = info.dims[axis];
  }

  return TensorView<T>(shape, reinterpret_cast<T *>(_inputs.at(index).data()));
}

template TensorView<float> IOManager::inputView<float>(uint32_t index);
template TensorView<int32_t> IOManager::inputView<int32_t>(uint32_t index);
template TensorView<int64_t> IOManager::inputView<int64_t>(uint32_t index);
template TensorView<bool> IOManager::inputView<bool>(uint32_t index);
template TensorView<uint8_t> IOManager::inputView<uint8_t>(uint32_t index);
template TensorView<int8_t> IOManager::inputView<int8_t>(uint32_t index);

std::vector<uint8_t> &IOManager::inputBase(uint32_t index)
{
  if (_inputs.find(index) == _inputs.end())
  {
    getInput(index);
  }

  return _inputs.at(index);
}

template <typename T> TensorView<T> IOManager::outputView(uint32_t index)
{
  if (_outputs.find(index) == _outputs.end())
  {
    getOutput(index);
  }

  assert(_output_infos.find(index) != _output_infos.end());
  auto &info = _output_infos.at(index);

  nnfw::misc::tensor::Shape shape(info.rank);
  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    shape.dim(axis) = info.dims[axis];
  }

  return TensorView<T>(shape, reinterpret_cast<T *>(_outputs.at(index).data()));
}

template TensorView<float> IOManager::outputView<float>(uint32_t index);
template TensorView<int32_t> IOManager::outputView<int32_t>(uint32_t index);
template TensorView<int64_t> IOManager::outputView<int64_t>(uint32_t index);
template TensorView<bool> IOManager::outputView<bool>(uint32_t index);
template TensorView<uint8_t> IOManager::outputView<uint8_t>(uint32_t index);
template TensorView<int8_t> IOManager::outputView<int8_t>(uint32_t index);

uint32_t IOManager::inputs() const
{
  uint32_t num_inputs = 0;
  NNFW_ASSERT_FAIL(nnfw_input_size(_session, &num_inputs),
                   "[ ERROR ] Failure during get input size");

  return num_inputs;
}

uint32_t IOManager::outputs() const
{
  uint32_t num_outputs = 0;
  NNFW_ASSERT_FAIL(nnfw_output_size(_session, &num_outputs),
                   "[ ERROR ] Failure during get output size");

  return num_outputs;
}

const nnfw_tensorinfo &IOManager::inputTensorInfo(uint32_t index)
{
  if (_inputs.find(index) == _inputs.end())
  {
    getInput(index);
  }

  return _input_infos.at(index);
}
const nnfw_tensorinfo &IOManager::outputTensorInfo(uint32_t index)
{
  if (_outputs.find(index) == _outputs.end())
  {
    getOutput(index);
  }

  return _output_infos.at(index);
}

void IOManager::setInput(uint32_t index)
{
  if (_inputs.find(index) == _inputs.end())
  {
    std::cerr << "[ ERROR ] Failure during set input #" << index << std::endl;
    exit(-1);
  }

  auto &info = _input_infos.at(index);

  NNFW_ASSERT_FAIL(
    nnfw_set_input(_session, index, info.dtype, _inputs.at(index).data(), sizeOfTensor(info)),
    "[ ERROR ] Failure to set input tensor buffer");
}

} // namespace onert_cmp
} // namespace nnfw
