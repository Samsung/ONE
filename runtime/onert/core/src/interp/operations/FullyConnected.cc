/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/operation/FullyConnected.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/FullyConnected.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace interp
{
namespace fc
{

void prepareFC(ExecEnv *env, const ir::Operation &node)
{
  const auto in_index = node.getInputs().at(ir::operation::FullyConnected::INPUT);
  const auto kernel_index = node.getInputs().at(ir::operation::FullyConnected::WEIGHT);
  const auto bias_index = node.getInputs().at(ir::operation::FullyConnected::BIAS);
  const auto out_index = node.getOutputs().at(0);

  const auto in_tensor = env->tensorAt(in_index);
  const auto kernel_tensor = env->tensorAt(kernel_index);
  const auto bias_tensor = env->tensorAt(bias_index);

  UNUSED_RELEASE(in_tensor);
  UNUSED_RELEASE(kernel_tensor);
  UNUSED_RELEASE(bias_tensor);

  assert(in_tensor->num_dimensions() >= 2);
  assert(kernel_tensor->num_dimensions() == 2);
  assert(bias_tensor->num_dimensions() == 1);

  const auto input_size_with_batch = in_tensor->num_elements();
  const auto num_units = kernel_tensor->dimension(0);
  const auto input_size = kernel_tensor->dimension(1);
  const auto batch_size = input_size_with_batch / input_size;
  assert(input_size_with_batch % input_size == 0);
  assert(num_units == bias_tensor->dimension(0));

  // Make output tensor info
  ir::Shape output_shape(2);
  output_shape.dim(0) = batch_size;
  output_shape.dim(1) = num_units;
  const auto out_info =
      ir::OperandInfo::createStaticInfo(output_shape, in_tensor->tensorInfo().typeInfo());
  env->allocateIfNeeded(out_index, out_info);

  auto out_tensor = env->tensorAt(out_index);
  UNUSED_RELEASE(out_tensor);

  // Handle same ifm & ofm data type only
  assert(in_tensor->data_type() == out_tensor->data_type());
  assert(out_tensor->num_dimensions() == 2);
  assert(out_tensor->dimension(0) == batch_size);
  assert(out_tensor->dimension(1) == num_units);
}

void invoke(const ITensor *ifm_tensor, const ITensor *ker_tensor, const ITensor *bias_tensor,
            const ITensor *ofm_tensor, const ir::operation::FullyConnected::Param &param)
{
  const auto ifm_buffer = ifm_tensor->bufferRO();
  const auto ker_buffer = ker_tensor->bufferRO();
  const auto bias_buffer = bias_tensor->bufferRO();
  auto ofm_buffer = ofm_tensor->buffer();

  // Calculate
  nnfw::cker::FullyConnectedParams cker_param;
  cker_param.activation = convertActivationType(param.activation);
  const auto cker_ifm_shape = convertShape(ifm_tensor->tensorInfo().shape());
  const auto cker_ker_shape = convertShape(ker_tensor->tensorInfo().shape());
  const auto cker_bias_shape = convertShape(bias_tensor->tensorInfo().shape());
  const auto cker_ofm_shape = convertShape(ofm_tensor->tensorInfo().shape());
  const float *ifm_ptr = reinterpret_cast<const float *>(ifm_buffer);
  const float *ker_ptr = reinterpret_cast<const float *>(ker_buffer);
  const float *bias_ptr = reinterpret_cast<const float *>(bias_buffer);
  float *ofm_ptr = reinterpret_cast<float *>(ofm_buffer);

  nnfw::cker::FullyConnected(cker_param, cker_ifm_shape, ifm_ptr, cker_ker_shape, ker_ptr,
                             cker_bias_shape, bias_ptr, cker_ofm_shape, ofm_ptr);
}

void invokeFC(const ExecEnv *env, const ir::Operation &node)
{
  const auto &conv_node =
      nnfw::misc::polymorphic_downcast<const ir::operation::FullyConnected &>(node);

  const auto ifm_index = node.getInputs().at(ir::operation::FullyConnected::INPUT);
  const auto ker_index = node.getInputs().at(ir::operation::FullyConnected::WEIGHT);
  const auto bias_index = node.getInputs().at(ir::operation::FullyConnected::BIAS);
  const auto ofm_index = node.getOutputs().at(0);

  const auto ifm_tensor = env->tensorAt(ifm_index);
  const auto ker_tensor = env->tensorAt(ker_index);
  const auto bias_tensor = env->tensorAt(bias_index);
  const auto ofm_tensor = env->tensorAt(ofm_index);

  const auto data_type = ifm_tensor->data_type();
  if (data_type == ir::DataType::FLOAT32)
  {
    invoke(ifm_tensor, ker_tensor, bias_tensor, ofm_tensor, conv_node.param());
  }
  else
  {
    throw std::runtime_error{"NYI: Support float only"};
  }
}
} // namespace fc

OpKernel *getFullyConnected()
{
  static OpKernel kernel = {fc::prepareFC, fc::invokeFC};
  return &kernel;
}

} // namespace interp
} // namespace onert
