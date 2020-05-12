/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Concatenation.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Concatenation::Concatenation(std::vector<const Tensor *> inputs, Tensor *output,
                             const ConcatenationParams &params)
    : _inputs(std::move(inputs)), _output(output), _params(params)
{
}

void Concatenation::configure()
{
  int32_t sum_axis = 0;
  for (const Tensor *input : _inputs)
  {
    sum_axis += input->shape().dim(_params.axis);
  }

  Shape output_shape = _inputs[0]->shape();
  output_shape.dim(_params.axis) = sum_axis;

  _output->resize(output_shape);
}

void Concatenation::execute() const
{
  switch (_inputs[0]->element_type())
  {
    case DataType::FLOAT32:
      evalGeneric<float>();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    case DataType::S8:
      evalGeneric<int8_t>();
      break;
    case DataType::S32:
      evalGeneric<int32_t>();
      break;
    case DataType::S64:
      evalGeneric<int64_t>();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T> void Concatenation::evalGeneric() const
{
  int axis = _params.axis;
  if (axis < 0)
    axis += _output->shape().num_dims();

  VectorOfTensors<T> inputs(_inputs);
  tflite::ConcatenationParams params{};
  params.axis = axis;
  params.inputs_count = _inputs.size();
  tflite::reference_ops::Concatenation(params, inputs.shapes(), inputs.data(),
                                       convertShape(_output->shape()), _output->data<T>());
}

void Concatenation::evalQuantized() const
{
  int axis = _params.axis;
  if (axis < 0)
    axis += _output->shape().num_dims();

  VectorOfQuantizedTensors inputs(_inputs);
  tflite::ConcatenationParams params{};
  params.axis = axis;
  params.input_zeropoint = inputs.zero_point();
  params.input_scale = inputs.scale();
  params.inputs_count = _inputs.size();
  params.output_zeropoint = _output->zero_point();
  params.output_scale = _output->scale();

  tflite::reference_ops::ConcatenationWithScaling(params, inputs.shapes(), inputs.data(),
                                                  convertShape(_output->shape()),
                                                  _output->data<uint8_t>());
}

} // namespace kernels
} // namespace luci_interpreter
