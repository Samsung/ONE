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

#include "moco/Pass/Passes/ResolveConstantShape.h"

#include <moco/Support/TFShapeInferenceHelper.h>
#include <moco/Support/NodeAs.h>

#include <moco/IR/Nodes/TFShape.h>
#include <moco/IR/Nodes/TFConst.h>

#include <loco.h>

#include <oops/UserExn.h>

#include <cassert>

namespace
{

/**
 * WHEN:
 *   - TFShape's input shape is determined
 * DO:
 *   - Replace TFShape into TFConst
 *
 *
 * <Before>
 *     in ---- TFShape ---- out(s)
 *
 * <After>
 *     in ---- TFShape
 *
 *             TFConst ---- out(s)
 */
bool resolve_constant_shape(loco::Graph *graph, moco::TFShape *shape_node)
{
  auto input_shape = moco::node_shape(shape_node->input());

  // Check condition
  if (input_shape.domain() == loco::Domain::Unknown)
  {
    // Cannot resolve without known input_shape
    return false;
  }

  auto input_tensor_shape = input_shape.as<loco::TensorShape>();

  auto shape_rank = input_tensor_shape.rank();
  for (uint32_t axis = 0; axis < shape_rank; ++axis)
  {
    if (!input_tensor_shape.dim(axis).known())
    {
      // Cannot resolve with unknown dimension
      return false;
    }
  }

  // Make TFConst to replace TFShape
  auto const_node = graph->nodes()->create<moco::TFConst>();

  // set dtype
  auto dtype = shape_node->dtype();
  const_node->dtype(dtype);

  // set shape
  const_node->rank(1);
  const_node->dim(0) = shape_rank;

  // set data
  if (dtype == loco::DataType::S32)
  {
    // TODO Better to make template for this when support new dtype
    const_node->size<loco::DataType::S32>(shape_rank);
    for (uint32_t axis = 0; axis < shape_rank; ++axis)
    {
      int32_t dim = (int32_t)input_tensor_shape.dim(axis).value();
      if (!(dim > 0))
      {
        throw oops::UserExn("Invalid input shape", shape_node->name());
      }
      const_node->at<loco::DataType::S32>(axis) = dim;
    }
  }
  else
  {
    throw oops::UserExn("Unsupported data type", shape_node->name());
  }

  // replace
  loco::replace(shape_node).with(const_node);

  return true;
}

} // namespace

namespace moco
{

bool ResolveConstantShape::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (auto shape_node = as<moco::TFShape>(node))
    {
      if (resolve_constant_shape(graph, shape_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace moco
