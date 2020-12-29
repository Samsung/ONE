/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/CircleShapeInferenceHelper.h"

#include <oops/InternalExn.h>

namespace luci
{

namespace sinf
{

loco::TensorShape circle_shape(const luci::CircleNode *node)
{
  loco::TensorShape shape;
  shape.rank(node->rank());
  for (uint32_t r = 0; r < node->rank(); ++r)
    shape.dim(r) = node->dim(r);
  return shape;
}

loco::TensorShape input_arg_shape(const luci::CircleNode *node, unsigned int index)
{
  if (node->arity() <= index)
    throw std::runtime_error("Arity index out of range");

  auto input_node = loco::must_cast<luci::CircleNode *>(node->arg(index));
  return circle_shape(input_node);
}

loco::TensorShape reduced_shape(const loco::Node *node, const loco::Node *indices, bool keep_dims)
{
  const auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
  const auto input_shape = circle_shape(circle_node);

  // If input shape is scalar, output shape is always scalar.
  // If input shape is not inferred, output shape cannot be inferred neither.
  // Therefore, if rank of input shape is 0, return empty shape.
  if (input_shape.rank() == 0)
  {
    return loco::TensorShape();
  }

  loco::TensorShape output_shape;

  // When reduction_indices is not constant
  auto reduction_indices = dynamic_cast<const luci::CircleConst *>(indices);
  if (reduction_indices == nullptr)
  {
    if (keep_dims)
    {
      // If keep_dims is true, rank is not changed.
      output_shape.rank(input_shape.rank());
      for (uint32_t i = 0; i < output_shape.rank(); ++i)
        output_shape.dim(i).unset();
    }
    else
    {
      // There is no way to inference for this case.
      // Return empty shape.
    }

    return output_shape;
  }

  std::vector<int32_t> reduction_values;
  if (reduction_indices->dtype() == loco::DataType::S32)
  {
    auto reduction_size = reduction_indices->size<loco::DataType::S32>();
    for (uint32_t i = 0; i < reduction_size; ++i)
    {
      int32_t axis = reduction_indices->at<loco::DataType::S32>(i);
      if (axis < 0)
        axis += input_shape.rank();

      if (!(0 <= axis && axis < static_cast<int32_t>(input_shape.rank())))
        INTERNAL_EXN_V("Invalid reduction axis for REDUCER", oops::to_uint32(axis));

      reduction_values.push_back(axis);
    }
  }
  else if (reduction_indices->dtype() == loco::DataType::S64)
  {
    auto reduction_size = reduction_indices->size<loco::DataType::S64>();
    for (uint32_t i = 0; i < reduction_size; ++i)
    {
      int32_t axis = static_cast<int32_t>(reduction_indices->at<loco::DataType::S64>(i));
      if (axis < 0)
        axis += input_shape.rank();

      if (!(0 <= axis && axis < static_cast<int32_t>(input_shape.rank())))
        INTERNAL_EXN_V("Invalid reduction axis for REDUCER", oops::to_uint32(axis));

      reduction_values.push_back(axis);
    }
  }
  else
  {
    INTERNAL_EXN("Wrong reduction axis type, Only INT32, INT64 supported.");
  }

  if (keep_dims)
  {
    output_shape.rank(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i)
      output_shape.dim(i) = input_shape.dim(i);
    for (uint32_t i = 0; i < reduction_values.size(); ++i)
      output_shape.dim(reduction_values.at(i)).set(1);
  }
  else
  {
    std::vector<bool> check_reduce(input_shape.rank(), false);
    for (uint32_t i = 0; i < reduction_values.size(); ++i)
      check_reduce.at(reduction_values.at(i)) = true;

    uint32_t reduce_cnt = 0;
    for (uint32_t i = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i))
        ++reduce_cnt;

    output_shape.rank(input_shape.rank() - reduce_cnt);
    for (uint32_t i = 0, j = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i) == false)
        output_shape.dim(j++) = input_shape.dim(i);
  }

  return output_shape;
}

} // namespace sinf

} // namespace luci
