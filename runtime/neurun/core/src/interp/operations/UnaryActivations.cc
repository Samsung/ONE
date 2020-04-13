/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cmath>

#include "OperationUtil.h"

#include "interp/Registration.h"

#include "ir/operation/ReLU.h"
#include "ir/operation/ReLU1.h"
#include "ir/operation/ReLU6.h"
#include "ir/operation/Tanh.h"

namespace neurun
{
namespace interp
{
namespace
{

enum class ActivationType
{
  ReLU,
  ReLU1,
  ReLU6,
  Tanh
};

void prepare(ExecEnv *env, const ir::Operation &node)
{
  const auto input_index = node.getInputs().at(0);
  const auto output_index = node.getOutputs().at(0);

  const auto input_tensor = env->tensorAt(input_index);

  const auto output_info = env->graph().operands().at(output_index).info();
  if (output_info.total_size() == 0)
  {
    // Output's shape and type is same with input
    auto input_info = input_tensor->tensorInfo();
    // We can handle already allocated (ex. model output)
    env->allocateIfNeeded(output_index, input_info);
  }
  else
  {
    env->allocateIfNeeded(output_index, output_info);
  }

  const auto output_tensor = env->tensorAt(output_index);
  // Check shape and type lhs is same with output
  // TODO Util function to compare TensorInfo
  if (input_tensor->data_type() != output_tensor->data_type())
  {
    throw std::runtime_error{"Interp(Activations): Invalid output type"};
  }
}

template <ActivationType act_type>
void evalFloat(const float *input_ptr, float *output_ptr, uint64_t num_elements)
{
  std::function<float(const float &)> fn = [](const float &) { return std::nanf(""); };
  switch (act_type)
  {
    case ActivationType::ReLU:
      fn = [](const float &in) { return std::max(0.f, in); };
      break;
    case ActivationType::ReLU1:
      fn = [](const float &in) { return std::min(std::max(-1.f, in), 1.f); };
      break;
    case ActivationType::ReLU6:
      fn = [](const float &in) { return std::min(std::max(0.f, in), 6.f); };
      break;
    case ActivationType::Tanh:
      fn = [](const float &in) { return std::tanh(in); };
      break;
    default:
      throw std::runtime_error{"Interp(Activations): NYI - Unsupported activation"};
      break;
  }

  const float *input_end = input_ptr + num_elements;
  for (; input_ptr < input_end; input_ptr++, output_ptr++)
  {
    *output_ptr = fn(*input_ptr);
  }
}

template <ActivationType act_type> void invoke(const ExecEnv *env, const ir::Operation &node)
{
  const auto input_index = node.getInputs().at(0);
  const auto output_index = node.getOutputs().at(0);

  // Check lhs shape is same with rhs (with broadcast)
  const auto input_tensor = env->tensorAt(input_index);
  const auto output_tensor = env->tensorAt(output_index);

  const auto data_type = input_tensor->data_type();
  if (data_type == ir::DataType::FLOAT32)
  {
    uint64_t elements = input_tensor->num_elements();
    const float *input_start = reinterpret_cast<const float *>(input_tensor->bufferRO());
    float *out = reinterpret_cast<float *>(output_tensor->buffer());

    evalFloat<act_type>(input_start, out, elements);
  }
  else
  {
    throw std::runtime_error{"Interp(ReLU6): NYI - Support float only"};
  }
}

} // namespace

OpKernel *getReLU()
{
  static OpKernel kernel = {prepare, invoke<ActivationType::ReLU>};
  return &kernel;
}

OpKernel *getReLU1()
{
  static OpKernel kernel = {prepare, invoke<ActivationType::ReLU1>};
  return &kernel;
}

OpKernel *getReLU6()
{
  static OpKernel kernel = {prepare, invoke<ActivationType::ReLU6>};
  return &kernel;
}

OpKernel *getTanh()
{
  static OpKernel kernel = {prepare, invoke<ActivationType::Tanh>};
  return &kernel;
}

} // namespace interp
} // namespace neurun
