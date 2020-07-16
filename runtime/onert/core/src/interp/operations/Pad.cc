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

#include <cker/operation/Pad.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/Pad.h"

namespace onert
{
namespace interp
{
namespace
{

void preparePad(ExecEnv *env, const ir::Operation &node)
{
  const auto input_index = node.getInputs().at(ir::operation::Pad::INPUT);
  const auto output_index = node.getOutputs().at(0);

  const auto input_tensor = env->tensorAt(input_index);

  const auto output_info = env->graph().operands().at(output_index).info();

  // Check shape and type lhs is same with rhs
  // TODO Util function to compare TensorInfo
  if (output_info.total_size() == 0)
  {
    throw std::runtime_error{"Interp(Pad): NYI unspecified output shape"};
  }
  else
  {
    env->allocateIfNeeded(output_index, output_info);
  }

  const auto output_tensor = env->tensorAt(output_index);
  if (input_tensor->data_type() != output_tensor->data_type())
  {
    throw std::runtime_error{"Interp(Pad): Invalid output type"};
  }
}

void invoke(const ITensor *input_tensor, const ITensor *pad_tensor, const ITensor *output_tensor)
{
  const auto input_buffer = input_tensor->bufferRO();
  const auto pad_buffer = pad_tensor->bufferRO();
  auto output_buffer = output_tensor->buffer();

  int32_t pad_rank = pad_tensor->dimension(0);

  const auto cker_input_shape = convertShape(input_tensor->tensorInfo().shape());
  const auto cker_output_shape = convertShape(output_tensor->tensorInfo().shape());
  const float *input_ptr = reinterpret_cast<const float *>(input_buffer);
  const int32_t *pad_ptr = reinterpret_cast<const int32_t *>(pad_buffer);
  float *output_ptr = reinterpret_cast<float *>(output_buffer);

  nnfw::cker::Pad<float>(pad_ptr, pad_rank, cker_input_shape, input_ptr, cker_output_shape,
                         output_ptr, nullptr);
}

void invokePad(const ExecEnv *env, const ir::Operation &node)
{
  const auto input_index = node.getInputs().at(ir::operation::Pad::INPUT);
  const auto pad_index = node.getInputs().at(ir::operation::Pad::PAD);
  const auto output_index = node.getOutputs().at(0);

  const auto input_tensor = env->tensorAt(input_index);
  const auto pad_tensor = env->tensorAt(pad_index);
  const auto output_tensor = env->tensorAt(output_index);

  const auto data_type = input_tensor->data_type();

  if (data_type == ir::DataType::FLOAT32)
  {
    invoke(input_tensor, pad_tensor, output_tensor);
  }
  else
  {
    throw std::runtime_error{"Interp(Pad): NYI - Unsupported data type"};
  }
}
} // namespace

OpKernel *getPad()
{
  static OpKernel kernel = {preparePad, invokePad};
  return &kernel;
}

} // namespace interp
} // namespace onert
