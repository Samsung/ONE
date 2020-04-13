/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Pass/Passes/ResolveReshapeWildcardDim.h"

#include <moco/Support/TFShapeInferenceHelper.h>
#include <moco/Support/NodeAs.h>

#include <moco/IR/Nodes/TFReshape.h>
#include <moco/IR/Nodes/TFConst.h>

#include <cassert>
#include <limits>

namespace
{

/**
 * @return  true  when 'node' has one and only one wildcard dimension
 * @return  false when 'node' has no wildcard dimension, i.e. fixed reshape case
 *
 * @note  Assertions in this function are sanity check for 'node', Reshape's
 *        Const shape input
 */
bool has_one_wildcard_dim(const moco::TFConst *node)
{
  assert(node->dtype() == loco::DataType::S32);
  assert(node->rank() == 1);

  auto len = node->dim(0).value();
  assert(len > 0);

  // Must have one and only wildcard dimension(-1)
  uint32_t count_wildcard_dim = 0;
  for (uint32_t i = 0; i < len; ++i)
  {
    auto dim = node->at<loco::DataType::S32>(i);
    if (dim == -1)
      count_wildcard_dim++;
    else
      assert(dim >= 1);
  }

  assert(count_wildcard_dim <= 1 &&
         "Invalid Reshape: there should be none or only one wildcard dimension");
  return count_wildcard_dim;
}

uint32_t volume(const loco::TensorShape &shape)
{
  uint32_t ret = 1;
  auto rank = shape.rank();
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    ret *= shape.dim(axis).value();
  }
  return ret;
}

void deduce_and_fix_wildcard_dim(moco::TFConst *node, const loco::NodeShape &tensor_input_shape)
{
  assert(has_one_wildcard_dim(node));

  assert(tensor_input_shape.domain() == loco::Domain::Tensor);
  auto shape = tensor_input_shape.as<loco::TensorShape>();

  auto len = node->dim(0).value();
  uint32_t wildcard_index = std::numeric_limits<uint32_t>::max();
  uint32_t product_of_non_wildcard_dims = 1;

  // Deduce
  for (uint32_t i = 0; i < len; ++i)
  {
    auto dim = node->at<loco::DataType::S32>(i);
    if (dim == -1)
    {
      wildcard_index = i;
    }
    else
    {
      product_of_non_wildcard_dims *= dim;
    }
  }
  assert(wildcard_index != std::numeric_limits<uint32_t>::max());

  // Fix
  assert(volume(shape) % product_of_non_wildcard_dims == 0);
  node->at<loco::DataType::S32>(wildcard_index) = volume(shape) / product_of_non_wildcard_dims;
}

/**
 * WHEN:
 *   - TFReshape's shape input is TFConst
 *   - The TFConst is valid shape input for dynamic reshape, i.e. it has one and
 *     only wildcard dimension(-1)
 *   - TFReshape's tensor input has complete shape inference data
 * DO:
 *   - Deduce what the wildcard dimension is and fix it
 */
bool resolve_wildcard_dim(moco::TFReshape *reshape)
{
  // Check conditions (WHEN)
  auto const_shape_input = dynamic_cast<moco::TFConst *>(reshape->shape());
  if (!const_shape_input)
    return false;

  if (!has_one_wildcard_dim(const_shape_input))
    return false;

  auto tensor_input_shape = moco::node_shape(reshape->tensor());
  if (tensor_input_shape.domain() == loco::Domain::Unknown)
    return false;

  // Deduce (DO)
  deduce_and_fix_wildcard_dim(const_shape_input, tensor_input_shape);

  return true;
}

} // namespace

namespace moco
{

bool ResolveReshapeWildcardDim::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (auto reshape = as<moco::TFReshape>(node))
    {
      if (resolve_wildcard_dim(reshape))
        changed = true;
    }
  }

  return changed;
}

} // namespace moco
