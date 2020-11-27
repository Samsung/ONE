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

#include <cker/operation/Gather.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/Gather.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace interp
{
namespace
{

void prepareGather(ExecEnv *env, const ir::Operation &node)
{
  const auto input_index = node.getInputs().at(ir::operation::Gather::INPUT);
  const auto indices_index = node.getInputs().at(ir::operation::Gather::INDICES);
  const auto output_index = node.getOutputs().at(0);

  const auto input_tensor = env->tensorAt(input_index);
  const auto indices_tensor = env->tensorAt(indices_index);

  // TODO handle unspecified output shape:
  //      calculate output shape using ifm shape, kernel shape, padding, stride
  const auto output_info = env->graph().operands().at(output_index).info();
  if (output_info.total_size() == 0)
  {
    throw std::runtime_error{"Interp(Gather): NYI for unspecified output shape"};
  }
  else
  {
    env->allocateIfNeeded(output_index, output_info);
  }

  if (indices_tensor->data_type() != ir::DataType::INT32)
  {
    throw std::runtime_error{"Interp(Gather): Invalid indices data type"};
  }

  auto output_tensor = env->tensorAt(output_index);
  auto output_rank = input_tensor->num_dimensions() + indices_tensor->num_dimensions() - 1;

  if (output_rank != output_tensor->num_dimensions())
  {
    throw std::runtime_error{"Interp(Gather): Invalid output rank"};
  }
  if (output_tensor->data_type() != input_tensor->data_type())
  {
    throw std::runtime_error{"Interp(Gather): Invalid output data type"};
  }

  if (input_tensor->data_type() == ir::DataType::QUANT_UINT8_ASYMM &&
      input_tensor->tensorInfo().typeInfo() != output_tensor->tensorInfo().typeInfo())
  {
    throw std::runtime_error{
      "Interp(Gather): Cannot handle different I/O QUANT_UINT8_ASYMM scale/offset"};
  }
}

template <typename raw_type>
void invoke(const ITensor *input_tensors, const ITensor *indices_tensors,
            const ITensor *output_tensor, uint32_t axis)
{
  // Calculate
  nnfw::cker::GatherParams cker_param;
  cker_param.axis = (int8_t)axis;

  const auto cker_input_shapes = convertShape(input_tensors->tensorInfo().shape());
  const auto cker_indices_shape = convertShape(indices_tensors->tensorInfo().shape());
  const auto cker_output_shape = convertShape(output_tensor->tensorInfo().shape());
  const raw_type *input_ptr = reinterpret_cast<const raw_type *>(input_tensors->bufferRO());
  const int32_t *indices_ptr = reinterpret_cast<const int32_t *>(indices_tensors->bufferRO());
  raw_type *output_ptr = reinterpret_cast<raw_type *>(output_tensor->buffer());

  nnfw::cker::Gather<raw_type>(cker_param, cker_input_shapes, input_ptr, cker_indices_shape,
                               indices_ptr, cker_output_shape, output_ptr);
}

void invokeGather(const ExecEnv *env, const ir::Operation &node)
{
  const auto &gather_node = nnfw::misc::polymorphic_downcast<const ir::operation::Gather &>(node);
  const int32_t axis_raw = gather_node.param().axis;

  const auto input_index = node.getInputs().at(ir::operation::Gather::INPUT);
  const auto indices_index = node.getInputs().at(ir::operation::Gather::INDICES);
  const auto output_index = node.getOutputs().at(0);

  const auto input_tensor = env->tensorAt(input_index);
  const auto indices_tensor = env->tensorAt(indices_index);
  const auto output_tensor = env->tensorAt(output_index);
  const uint32_t axis = (axis_raw < 0) ? (axis_raw + input_tensor->num_dimensions()) : axis_raw;

  const auto data_type = input_tensor->data_type();

  switch (data_type)
  {
    case ir::DataType::FLOAT32:
      invoke<float>(input_tensor, indices_tensor, output_tensor, axis);
      break;
    case ir::DataType::INT32:
      invoke<int32_t>(input_tensor, indices_tensor, output_tensor, axis);
      break;
    case ir::DataType::QUANT_UINT8_ASYMM:
      invoke<uint8_t>(input_tensor, indices_tensor, output_tensor, axis);
      break;
    default:
      throw std::runtime_error{"Interp(Gather): NYI - Not supported type"};
  }
}

} // namespace

OpKernel *getGather()
{
  static OpKernel kernel = {prepareGather, invokeGather};
  return &kernel;
}

} // namespace interp
} // namespace onert
