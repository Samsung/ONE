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

#include "luci/Service/CircleShapeSignatureInferenceHelper.h"

#include <loco.h>

#include <luci/Log.h>

#include <oops/InternalExn.h>

namespace
{

luci::ShapeSignature clean_signature(const luci::ShapeSignature &signature)
{
  // If shape signature has at least one -1, it is not static.
  for (uint32_t i = 0; i < signature.rank(); ++i)
    if (signature.dim(i) == -1)
      return signature;

  // If all dimensions are static, return empty shape signature.
  return luci::ShapeSignature();
}

} // namespace

namespace luci
{

namespace ssinf
{

ShapeSignature reduced_signature(const loco::Node *node, const loco::Node *indices, bool keep_dims)
{
  LOGGER(l);

  ShapeSignature input_signature;
  ShapeSignature output_signature;

  auto circle_node = loco::must_cast<const luci::CircleNode *>(node);

  if (circle_node->shape_signature().rank() > 0)
    input_signature = circle_node->shape_signature();
  else
  {
    input_signature.rank(circle_node->rank());
    for (uint32_t i = 0; i < circle_node->rank(); ++i)
      input_signature.dim(i) = circle_node->dim(i).value();
  }

  // When reduction_indices is not constant
  auto reduction_indices = dynamic_cast<const luci::CircleConst *>(indices);
  if (reduction_indices == nullptr)
  {
    if (keep_dims)
    {
      // If keep_dims is true, rank is not changed.
      output_signature.rank(input_signature.rank());
      for (uint32_t i = 0; i < output_signature.rank(); ++i)
        output_signature.dim(i) = -1;
    }
    else
    {
      // There is no way to inference for this case.
      // Do nothing to return empty signature.
      INFO(l) << "[CircleShapeSignatureInferenceHelper] " << circle_node->name() << std::endl;
      INFO(l) << " reduced_signature : cannot infer because of non-constant node" << std::endl;
    }

    return output_signature;
  }

  // If input rank is 0, it means that one of following case is occurred.
  // - Input is scalar : result is always scalar
  // - Input shape signature is not inferenced : cannot infer output shape signauture
  // Therefore, when input signature rank is 0, always return empty signature.
  if (input_signature.rank() == 0)
    return output_signature;

  std::vector<int32_t> reduction_values;
  if (reduction_indices->dtype() == loco::DataType::S32)
  {
    auto reduction_size = reduction_indices->size<loco::DataType::S32>();
    for (uint32_t i = 0; i < reduction_size; ++i)
    {
      int32_t axis = reduction_indices->at<loco::DataType::S32>(i);
      if (axis < 0)
        axis += input_signature.rank();

      if (!(0 <= axis && axis < static_cast<int32_t>(input_signature.rank())))
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
        axis += input_signature.rank();

      if (!(0 <= axis && axis < static_cast<int32_t>(input_signature.rank())))
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
    output_signature.rank(input_signature.rank());
    for (uint32_t i = 0; i < input_signature.rank(); ++i)
      output_signature.dim(i) = input_signature.dim(i);
    for (uint32_t i = 0; i < reduction_values.size(); ++i)
      output_signature.dim(reduction_values.at(i)) = 1;
  }
  else
  {
    std::vector<bool> check_reduce(input_signature.rank(), false);
    for (uint32_t i = 0; i < reduction_values.size(); ++i)
      check_reduce.at(reduction_values.at(i)) = true;

    uint32_t reduce_cnt = 0;
    for (uint32_t i = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i))
        ++reduce_cnt;

    output_signature.rank(input_signature.rank() - reduce_cnt);
    for (uint32_t i = 0, j = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i) == false)
        output_signature.dim(j++) = input_signature.dim(i);
  }

  return clean_signature(output_signature);
}

ShapeSignature input_arg_signature(const luci::CircleNode *node, uint32_t index)
{
  auto circle_input = loco::must_cast<luci::CircleNode *>(node->arg(index));
  return circle_input->shape_signature();
}

} // namespace ssinf

} // namespace luci
