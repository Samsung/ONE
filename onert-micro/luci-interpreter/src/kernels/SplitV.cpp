/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "SplitV.h"

#include "Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter
{
namespace kernels
{

SplitV::SplitV(const Tensor *input, const Tensor *size_splits, const Tensor *axis,
               std::vector<Tensor *> outputs)
  : Kernel({input, size_splits, axis}, std::move(outputs))
{
}

void SplitV::configure()
{
  assert(axis()->shape().num_elements() == 1);
  _axis_value = getTensorData<int32_t>(axis())[0];
  if (_axis_value < 0)
    _axis_value += input()->shape().num_dims();
  assert(_axis_value >= 0 && _axis_value < input()->shape().num_dims());

  auto num_split = static_cast<int32_t>(_outputs.size());
  auto sizes_data = getTensorData<int32_t>(size_splits());

  assert(size_splits()->shape().num_dims() == 1);

  int32_t sum = 0;
  const auto num_dims_size_spits = size_splits()->shape().dim(0);
  int32_t count_neg_dim = 0;

  for (int32_t i = 0; i < num_dims_size_spits - 1; ++i)
  {
    if (sizes_data[i] != -1)
    {
      sum += sizes_data[i];
    }
    else
    {
      count_neg_dim++;
    }
  }
  assert(count_neg_dim < 2);
  assert(size_splits()->shape().num_elements() == num_split);

  // TODO: enable it only if kernel with dynamic shapes
  auto output_shape = input()->shape();
  for (int32_t i = 0; i < num_split; ++i)
  {
    if (sizes_data[i] == -1)
    {
      output_shape.dim(_axis_value) = input()->shape().dim(_axis_value) - sum;
    }
    else
    {
      output_shape.dim(_axis_value) = sizes_data[i];
    }
    _outputs[i]->resize(output_shape);
  }
}

void SplitV::execute() const
{
  tflite::SplitParams params{};
  params.num_split = _outputs.size();
  params.axis = _axis_value;

#define TF_LITE_SPLIT(scalar)                                                                     \
  {                                                                                               \
    VectorOfTensors<scalar, false> all_outputs(_outputs);                                         \
    tflite::optimized_ops::Split(params, getTensorShape(input()), getTensorData<scalar>(input()), \
                                 all_outputs.shapes(), all_outputs.data());                       \
  }

  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      TF_LITE_SPLIT(float);
      break;
    case DataType::U8:
      TF_LITE_SPLIT(uint8_t);
      break;
    case DataType::S16:
      TF_LITE_SPLIT(int16_t);
      break;
    default:
      assert(false && "Unsupported type.");
  }
#undef TF_LITE_SPLIT
}

} // namespace kernels
} // namespace luci_interpreter
