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

#include <cker/operation/Concatenation.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/Concat.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace interp
{
namespace concat
{

void prepareConcat(ExecEnv *env, const ir::Operation &node)
{
  const auto &concat_node = nnfw::misc::polymorphic_downcast<const ir::operation::Concat &>(node);

  const auto first_index = node.getInputs().at(0);
  const auto out_index = node.getOutputs().at(0);

  const auto first_tensor = env->tensorAt(first_index);
  uint32_t out_axis_dimension = 0;
  const int32_t axis_raw = concat_node.param().axis;
  const uint32_t axis = (axis_raw < 0) ? (axis_raw + first_tensor->num_dimensions()) : axis_raw;

  // All inputs shape should be same except axis dimension
  // All inputs type should be same
  for (auto input : node.getInputs())
  {
    assert(first_tensor->num_dimensions() == env->tensorAt(input)->num_dimensions());
    assert(first_tensor->data_type() == env->tensorAt(input)->data_type());
    for (uint32_t i = 0; i < first_tensor->num_dimensions(); i++)
    {
      if (i == axis)
      {
        out_axis_dimension += env->tensorAt(input)->dimension(i);
        continue;
      }
      assert(first_tensor->dimension(i) == env->tensorAt(input)->dimension(i));
    }
  }

  // Make output tensor info using first input tensor info, and accumulated axis dimension value
  auto out_shape = first_tensor->tensorInfo().shape();
  out_shape.dim(axis) = out_axis_dimension;
  env->allocateIfNeeded(
    out_index, ir::OperandInfo::createStaticInfo(out_shape, first_tensor->tensorInfo().typeInfo()));

  auto out_tensor = env->tensorAt(out_index);
  UNUSED_RELEASE(out_tensor);

  // Output shape should be same with input except axis dimension
  // Output type should be same with input
  assert(first_tensor->data_type() == out_tensor->data_type());
  for (uint32_t i = 0; i < first_tensor->num_dimensions(); i++)
  {
    if (i == axis)
    {
      continue;
    }
    assert(first_tensor->dimension(i) == out_tensor->dimension(i));
  }
}

void invoke(const std::vector<const ITensor *> in_tensors, const ITensor *out_tensor, uint32_t axis)
{
  const uint32_t count = in_tensors.size();

  // Calculate
  nnfw::cker::ConcatenationParams cker_param;
  cker_param.axis = (int8_t)axis;
  cker_param.inputs_count = count;

  const auto out_shape = convertShape(out_tensor->tensorInfo().shape());

  std::vector<nnfw::cker::Shape> in_shapes;
  std::vector<const nnfw::cker::Shape *> in_shape_ptrs;
  in_shapes.reserve(count);
  in_shape_ptrs.reserve(count);
  std::vector<const float *> in_ptrs;
  for (uint32_t i = 0; i < count; i++)
  {
    in_shapes.push_back(convertShape(in_tensors[i]->tensorInfo().shape()));
    in_shape_ptrs.push_back(&in_shapes[i]);
    in_ptrs.push_back(reinterpret_cast<const float *>(in_tensors[i]->bufferRO()));
  }

  auto out_buffer = out_tensor->buffer();
  float *out_ptr = reinterpret_cast<float *>(out_buffer);

  nnfw::cker::Concatenation<float>(cker_param, in_shape_ptrs.data(), in_ptrs.data(), out_shape,
                                   out_ptr);
}

void invokeConcat(const ExecEnv *env, const ir::Operation &node)
{
  const auto &concat_node = nnfw::misc::polymorphic_downcast<const ir::operation::Concat &>(node);
  const int32_t axis_raw = concat_node.param().axis;

  std::vector<const ITensor *> in_tensors;
  for (const auto &e : concat_node.getInputs())
  {
    in_tensors.emplace_back(env->tensorAt(e));
  }

  const auto out_index = node.getOutputs().at(0);
  const auto out_tensor = env->tensorAt(out_index);
  const uint32_t axis = (axis_raw < 0) ? (axis_raw + out_tensor->num_dimensions()) : axis_raw;

  const auto data_type = in_tensors[0]->data_type();
  if (data_type == ir::DataType::FLOAT32)
  {
    invoke(in_tensors, out_tensor, axis);
  }
  else
  {
    throw std::runtime_error{"NYI: Support float32 only"};
  }
}
} // namespace concat

OpKernel *getConcat()
{
  static OpKernel kernel = {concat::prepareConcat, concat::invokeConcat};
  return &kernel;
}

} // namespace interp
} // namespace onert
