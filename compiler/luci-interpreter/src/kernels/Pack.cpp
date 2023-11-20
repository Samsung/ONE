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

#include "kernels/Pack.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Pack::Pack(std::vector<const Tensor *> inputs, Tensor *output, const PackParams &params)
  : KernelWithParams<PackParams>(std::move(inputs), {output}, params)
{
}

void Pack::configure()
{
  LUCI_INTERPRETER_CHECK(_inputs.size() == static_cast<uint32_t>(params().values_count));
  const Tensor *t0 = _inputs[0];
  const int dimension_size = t0->shape().num_dims() + 1;
  int axis = params().axis;
  if (axis < 0)
  {
    axis += dimension_size;
  }
  LUCI_INTERPRETER_CHECK(axis >= 0 && axis <= t0->shape().num_dims());

  if (t0->element_type() != DataType::S32 && t0->element_type() != DataType::FLOAT32 &&
      t0->element_type() != DataType::U8 && t0->element_type() != DataType::S8 &&
      t0->element_type() != DataType::S16 && t0->element_type() != DataType::S64)
  {
    throw std::runtime_error("luci-intp Pack(1) Unsupported type.");
  }

  for (uint32_t i = 1; i < _inputs.size(); ++i)
  {
    const Tensor *tensor = _inputs[i];
    LUCI_INTERPRETER_CHECK(tensor->element_type() == t0->element_type());
    LUCI_INTERPRETER_CHECK(tensor->shape().num_dims() == t0->shape().num_dims());
    for (int d = 0; d < t0->shape().num_dims(); ++d)
    {
      LUCI_INTERPRETER_CHECK(tensor->shape().dim(d) == t0->shape().dim(d));
    }
  }

  Shape output_shape(dimension_size);
  int i = 0;
  for (int index = 0; index < dimension_size; ++index)
  {
    if (index == axis)
    {
      output_shape.dim(index) = params().values_count;
    }
    else
    {
      output_shape.dim(index) = t0->shape().dim(i++);
    }
  }

  if (t0->element_type() == DataType::U8 || t0->element_type() == DataType::S8 ||
      t0->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(output()->zero_point() == t0->zero_point());
    LUCI_INTERPRETER_CHECK(output()->scale() == t0->scale());
    // Guarantee input/output quantization params match as we do not support
    // packing quantized tensors.
    for (int i = 0; i < params().values_count; i++)
    {
      LUCI_INTERPRETER_CHECK(_inputs[i]->zero_point() == t0->zero_point());
      LUCI_INTERPRETER_CHECK(_inputs[i]->scale() == t0->scale());
    }
  }

  output()->resize(output_shape);
}

void Pack::execute() const
{
  switch (_inputs[0]->element_type())
  {
    case DataType::FLOAT32:
      evalGeneric<float>();
      break;
    case DataType::U8:
      evalGeneric<uint8_t>();
      break;
    case DataType::S8:
      evalGeneric<int8_t>();
      break;
    case DataType::S16:
      evalGeneric<int16_t>();
      break;
    case DataType::S32:
      evalGeneric<int32_t>();
      break;
    case DataType::S64:
      evalGeneric<int64_t>();
      break;
    default:
      throw std::runtime_error("luci-intp Pack(2) Unsupported type.");
  }
}

template <typename T> void Pack::evalGeneric() const
{
  const Tensor *t0 = _inputs[0];
  const int dimension_size = t0->shape().num_dims() + 1;
  int axis = params().axis;
  if (axis < 0)
  {
    axis += dimension_size;
  }

  VectorOfTensors<T, true> inputs(_inputs);
  tflite::PackParams params{};
  params.axis = axis;
  params.inputs_count = _inputs.size();
  tflite::reference_ops::Pack<T>(params, inputs.shapes(), inputs.data(), getTensorShape(output()),
                                 getTensorData<T>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
