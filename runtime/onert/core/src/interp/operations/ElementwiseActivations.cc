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

#include "ir/operation/ElementwiseActivation.h"

#include <misc/polymorphic_downcast.h>
#include <cker/operation/Logistic.h>
#include <cker/operation/Tanh.h>

namespace onert
{
namespace interp
{
namespace
{

enum class ActivationType
{
  Logistic,
  ReLU,
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
    throw std::runtime_error{"Interp(ElementwiseActivation): Invalid output type"};
  }
}

template <ActivationType act_type>
void evalFloat(const float *input_ptr, float *output_ptr, uint64_t num_elements, float alpha,
               float beta)
{
  std::function<float(const float &)> fn = [](const float &) { return std::nanf(""); };
  switch (act_type)
  {
    case ActivationType::ReLU:
      fn = [alpha, beta](const float &in) { return std::min(std::max(beta, in), alpha); };
      break;
    case ActivationType::Tanh:
      fn = [](const float &in) { return std::tanh(in); };
      break;
    default:
      throw std::runtime_error{"Interp(ElementwiseActivation): NYI - Unsupported activation"};
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
    if (act_type == ActivationType::Logistic)
    {
      const auto cker_input_shape = convertShape(input_tensor->tensorInfo().shape());
      const auto cker_output_shape = convertShape(output_tensor->tensorInfo().shape());
      nnfw::cker::Logistic(cker_input_shape, input_start, cker_output_shape, out);
    }
    else
    {
      const auto &act_node =
        nnfw::misc::polymorphic_downcast<const ir::operation::ElementwiseActivation &>(node);
      evalFloat<act_type>(input_start, out, elements, act_node.param().alpha,
                          act_node.param().beta);
    }
  }
  else
  {
    throw std::runtime_error{"Interp(" + node.name() + "): NYI - Support float only"};
  }
}

void invokeElementwiseActivation(const ExecEnv *env, const ir::Operation &node)
{
  const auto &act_node =
    nnfw::misc::polymorphic_downcast<const ir::operation::ElementwiseActivation &>(node);
  switch (act_node.param().op_type)
  {
    case ir::operation::ElementwiseActivation::Type::LOGISTIC:
      invoke<ActivationType::Logistic>(env, node);
      break;
    case ir::operation::ElementwiseActivation::Type::RELU:
      invoke<ActivationType::ReLU>(env, node);
      break;
    case ir::operation::ElementwiseActivation::Type::TANH:
      invoke<ActivationType::Tanh>(env, node);
      break;
    default:
      throw std::runtime_error("Interp(" + node.name() + "): NYI - Unsupported activation");
  }
}

} // namespace

OpKernel *getElementwiseActivation()
{
  static OpKernel kernel = {prepare, invokeElementwiseActivation};
  return &kernel;
}

} // namespace interp
} // namespace onert
