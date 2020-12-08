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

#include <cker/operation/InstanceNorm.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/InstanceNorm.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace interp
{
namespace instancenorm
{

void prepareInstanceNorm(ExecEnv *env, const ir::Operation &node)
{
  const auto &instancenorm_node =
    nnfw::misc::polymorphic_downcast<const ir::operation::InstanceNorm &>(node);

  const auto input_index = node.getInputs().at(instancenorm_node.INPUT);
  const auto output_index = node.getOutputs().at(0);
  const auto input_tensor = env->tensorAt(input_index);

  if (input_tensor->num_dimensions() != 4)
  {
    throw std::runtime_error{"Interp(InstanceNorm): Input should be 4D-tensor"};
  }

  // Output shape should be same with input
  env->allocateIfNeeded(output_index, input_tensor->tensorInfo());

  auto output_tensor = env->tensorAt(output_index);
  UNUSED_RELEASE(output_tensor);

  // Handle same ifm & ofm data type only
  assert(input_tensor->data_type() == output_tensor->data_type());
  assert(input_tensor->tensorInfo().shape() == output_tensor->tensorInfo().shape());
}

inline void setActivationParams(float min, float max, nnfw::cker::InstanceNormParams *params)
{
  params->float_activation_min = min;
  params->float_activation_max = max;
}

void invoke(const ITensor *input_tensor, const ITensor *gamma_tensor, const ITensor *beta_tensor,
            const ITensor *output_tensor, const ir::operation::InstanceNorm::Param &param)
{
  // Calculate
  float activation_min, activation_max;
  calculateActivationRange(param.activation, &activation_min, &activation_max);

  nnfw::cker::InstanceNormParams cker_param;
  cker_param.epsilon = param.epsilon;
  cker_param.float_activation_min = activation_min;
  cker_param.float_activation_max = activation_max;

  const auto cker_input_shape = convertShape(input_tensor->tensorInfo().shape());
  const auto cker_gamma_shape = convertShape(gamma_tensor->tensorInfo().shape());
  const auto cker_beta_shape = convertShape(beta_tensor->tensorInfo().shape());
  const auto cker_output_shape = convertShape(output_tensor->tensorInfo().shape());
  const float *input_ptr = reinterpret_cast<const float *>(input_tensor->bufferRO());
  const float *gamma_ptr = reinterpret_cast<const float *>(gamma_tensor->bufferRO());
  const float *beta_ptr = reinterpret_cast<const float *>(beta_tensor->bufferRO());
  float *output_ptr = reinterpret_cast<float *>(output_tensor->buffer());

  nnfw::cker::InstanceNorm(cker_param, cker_input_shape, input_ptr, cker_gamma_shape, gamma_ptr,
                           cker_beta_shape, beta_ptr, cker_output_shape, output_ptr);
}

void invokeInstanceNorm(const ExecEnv *env, const ir::Operation &node)
{
  const auto &instancenorm_node =
    nnfw::misc::polymorphic_downcast<const ir::operation::InstanceNorm &>(node);

  const auto input_index = node.getInputs().at(instancenorm_node.INPUT);
  const auto gamma_index = node.getInputs().at(instancenorm_node.GAMMA);
  const auto beta_index = node.getInputs().at(instancenorm_node.BETA);
  const auto out_index = node.getOutputs().at(0);
  const auto input_tensor = env->tensorAt(input_index);
  const auto gamma_tensor = env->tensorAt(gamma_index);
  const auto beta_tensor = env->tensorAt(beta_index);
  const auto out_tensor = env->tensorAt(out_index);
  const auto data_type = input_tensor->data_type();

  if (data_type == ir::DataType::FLOAT32)
  {
    invoke(input_tensor, gamma_tensor, beta_tensor, out_tensor, instancenorm_node.param());
  }
  else
  {
    throw std::runtime_error{"NYI: Unsupported data type"};
  }
}
} // namespace instancenorm

OpKernel *getInstanceNorm()
{
  static OpKernel kernel = {instancenorm::prepareInstanceNorm, instancenorm::invokeInstanceNorm};
  return &kernel;
}

} // namespace interp
} // namespace onert
