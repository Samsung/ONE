/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Builders.h"
#include "Utils.h"

namespace luci_interpreter
{
namespace
{

template <typename T>
void splitImpl(const circle::Operator *cur_op, const circle::Tensor *input, int axis_value,
               BaseRuntimeGraph *runtime_graph)
{
  const int output_count = cur_op->outputs()->size();

  const auto output0_index = cur_op->outputs()->operator[](0);
  assert(output0_index != -1);

  const auto output0 = runtime_graph->getCircleTensorByIndex(output0_index);
  assert(output0 != nullptr);

  const int split_dimensions = Tensor::num_dims(input);
  int axis = axis_value < 0 ? axis_value + split_dimensions : axis_value;

  assert(axis < split_dimensions);
  assert(Tensor::num_dims(output0) == split_dimensions);

  int64_t split_size = Tensor::dim(output0, axis) * output_count;
  assert(split_size == Tensor::dim(input, axis));

  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= Tensor::dim(input, i);
  }

  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i)
  {
    base_inner_size *= Tensor::dim(input, i);
  }

  const T *input_ptr = kernels::getTensorData<T>(runtime_graph->getDataByTensor(input));
  for (int k = 0; k < outer_size; ++k)
  {
    for (int i = 0; i < output_count; ++i)
    {
      const auto output_index = cur_op->outputs()->operator[](i);
      assert(output_index != -1);

      const auto output = runtime_graph->getCircleTensorByIndex(output_index);
      assert(output != nullptr);

      T *output_data = kernels::getTensorData<T>(runtime_graph->getDataByTensor(output));
      const int copy_size = Tensor::dim(output0, axis) * base_inner_size;
      T *output_ptr = output_data + k * copy_size;
      for (int j = 0; j < copy_size; ++j)
        output_ptr[j] = input_ptr[j];
      input_ptr += copy_size;
    }
  }
}

} // namespace

void configure_kernel_CircleSplit(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto axis_index = cur_op->inputs()->operator[](1);

  LUCI_INTERPRETER_CHECK(input_index != -1);
  LUCI_INTERPRETER_CHECK(axis_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto axis = runtime_graph->getCircleTensorByIndex(axis_index);

  LUCI_INTERPRETER_CHECK(input != nullptr);
  LUCI_INTERPRETER_CHECK(axis != nullptr);
}

void execute_kernel_CircleSplit(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
                                bool)
{
  const auto input_index = cur_op->inputs()->operator[](1);
  const auto axis_index = cur_op->inputs()->operator[](0);

  assert(input_index != -1);
  assert(axis_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto axis = runtime_graph->getCircleTensorByIndex(axis_index);

  assert(input != nullptr);
  assert(axis != nullptr);

  const auto *axis_data = runtime_graph->getDataByTensor(axis);
  if (axis_data == nullptr)
    axis_data = runtime_graph->getConstDataByTensor(axis);

  assert(axis_data);

  int32_t axis_value = (kernels::getTensorData<int32_t>(axis_data))[0];
  if (axis_value < 0)
    axis_value += Tensor::num_dims(input);

  assert(axis_value >= 0);
  assert(axis_value < Tensor::num_dims(input));

  switch (Tensor::element_type(input))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
    {
      return splitImpl<float>(cur_op, input, axis_value, runtime_graph);
    }
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case DataType::S8:
    {
      return splitImpl<int8_t>(cur_op, input, axis_value, runtime_graph);
    }
    case DataType::S16:
    {
      return splitImpl<int16_t>(cur_op, input, axis_value, runtime_graph);
    }
#endif // DIS_QUANT
    case DataType::S32:
    {
      return splitImpl<int32_t>(cur_op, input, axis_value, runtime_graph);
    }
    default:
      assert(false && "Unsupported type");
  }
}

} // namespace luci_interpreter
