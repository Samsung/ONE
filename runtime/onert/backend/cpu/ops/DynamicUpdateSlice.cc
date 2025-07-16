/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DynamicUpdateSlice.h"
#include "OperationUtils.h"

namespace onert::backend::cpu::ops
{

template <typename T>
void update_slice(int32_t current_dim, int32_t max_dim, const std::vector<int32_t> output_stride,
                  const std::vector<int32_t> update_stride, const ir::Shape &update_shape,
                  const T *update, const std::vector<int64_t> indices_data, T *output)
{
  if (current_dim == max_dim)
    return;

  if (current_dim == max_dim - 1)
  {
    output += indices_data[current_dim] * output_stride[current_dim];
    memcpy(output, update, update_shape.dim(max_dim - 1) * sizeof(T));
  }
  else
  {
    output += indices_data[current_dim] * output_stride[current_dim];
    for (int i = 0; i < update_shape.dim(current_dim); ++i)
    {
      update_slice(current_dim + 1, max_dim, output_stride, update_stride, update_shape, update,
                   indices_data, output);
      output += output_stride[current_dim];
      update += update_stride[current_dim];
    }
  }
}

template <typename T>
void dynamicUpdateSlice(const IPortableTensor *input, const IPortableTensor *update,
                        const std::vector<int64_t> &indices_data, IPortableTensor *output)
{
  const auto &input_shape = input->get_info().shape();
  const auto &update_shape = update->get_info().shape();
  const T *input_data = getBuffer<T>(input);
  const T *update_data = getBuffer<T>(update);
  T *output_data = getBuffer<T>(output);

  // Special case 1 : output is copy of update
  if (input_shape == update_shape)
  {
    memcpy(output_data, update_data, update->get_info().total_size());
    return;
  }

  // Prepare update
  if (input_data != output_data)
    memcpy(output_data, input_data, input->get_info().total_size());

  // Special case 2: no update
  if (update_shape.num_elements() == 0)
    return;

  // Calculate clamped_start_indices
  const auto input_dims = input_shape.rank();
  std::vector<int64_t> clamped_start_indices(input_dims, 0);
  for (int i = 0; i < input_dims; i++)
  {
    clamped_start_indices[i] = std::min<int64_t>(std::max<int64_t>(0, indices_data[i]),
                                                 input_shape.dim(i) - update_shape.dim(i));
  }

  // Calculate strides
  std::vector<int32_t> output_stride(input_dims);
  std::vector<int32_t> update_stride(input_dims);
  output_stride[input_dims - 1] = 1;
  update_stride[input_dims - 1] = 1;
  for (int i = input_dims - 2; i >= 0; --i)
  {
    output_stride[i] = output_stride[i + 1] * input_shape.dim(i + 1);
    update_stride[i] = update_stride[i + 1] * update_shape.dim(i + 1);
  }

  update_slice<T>(0, input_dims, output_stride, update_stride, update_shape, update_data,
                  clamped_start_indices, output_data);
}

DynamicUpdateSliceLayer::DynamicUpdateSliceLayer()
  : _operand(nullptr), _update(nullptr), _start_indices(nullptr), _output(nullptr)
{
  // DO NOTHING
}

DynamicUpdateSliceLayer::~DynamicUpdateSliceLayer() = default;

void DynamicUpdateSliceLayer::configure(const IPortableTensor *operand,
                                        const IPortableTensor *update,
                                        const IPortableTensor *start_indices,
                                        IPortableTensor *output)
{
  assert(operand != nullptr);
  assert(update != nullptr);
  assert(start_indices != nullptr);
  assert(output != nullptr);

  _operand = operand;
  _update = update;
  _start_indices = start_indices;
  _output = output;
}

void DynamicUpdateSliceLayer::run()
{
  // Get indices data as int64 type vector
  std::vector<int64_t> indices_data(_start_indices->getShape().num_elements());
  for (size_t i = 0; i < indices_data.size(); ++i)
  {
    if (_start_indices->data_type() == OperandType::INT32)
    {
      indices_data[i] = static_cast<int64_t>(getBuffer<int32_t>(_start_indices)[i]);
    }
    else
    {
      assert(_start_indices->data_type() == OperandType::INT64);
      indices_data[i] = getBuffer<int64_t>(_start_indices)[i];
    }
  }

  switch (_operand->data_type())
  {
    case OperandType::FLOAT32:
      dynamicUpdateSlice<float>(_operand, _update, indices_data, _output);
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      dynamicUpdateSlice<uint8_t>(_operand, _update, indices_data, _output);
      break;
    case OperandType::QUANT_INT16_SYMM:
      dynamicUpdateSlice<int8_t>(_operand, _update, indices_data, _output);
      break;
    default:
      throw std::runtime_error{"DynamicUpdateSlice: NYI - unsupported data type"};
      break;
  }
}

} // namespace onert::backend::cpu::ops
