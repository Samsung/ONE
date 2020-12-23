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

#include <logo/ConstantFoldingPass.h>

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>

#include <stdex/Memory.h>

#include <locomotiv/Session.h>

#include <cassert>
#include <stdexcept>

namespace
{

uint64_t num_elements(const loco::NodeMixin<loco::NodeTrait::TensorShape> &shape)
{
  if (shape.rank() == 0)
  {
    return 0;
  }

  uint64_t res = 1;

  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    assert(shape.dim(axis).known());
    res *= shape.dim(axis).value();
  }

  return res;
}

/// @brief For some op, constant folding should not be performed. This returns true if node is such
/// op.
bool skip(const loco::Node *node)
{
  static std::set<uint32_t> skip_op = {
      // TODO Current implementation works for 'Tensor' domain only. Support other domains such as
      //      `Feature`, `Filter`, `Bias`, etc.
      static_cast<uint32_t>(loco::CanonicalOpcode::FilterEncode),
      static_cast<uint32_t>(loco::CanonicalOpcode::FeatureEncode),
      static_cast<uint32_t>(loco::CanonicalOpcode::BiasEncode),
      static_cast<uint32_t>(loco::CanonicalOpcode::DepthwiseFilterEncode),

      // We don't perform constant folding for Push
      static_cast<uint32_t>(loco::CanonicalOpcode::Push),

      // TensorBroadcast is a good hint for optimization
      // TODO Let this option be controlled by driver using logo
      static_cast<uint32_t>(loco::CanonicalOpcode::TensorBroadcast),
  };

  if (node->dialect() == loco::CanonicalDialect::get())
  {
    if (skip_op.find(node->opnum()) != skip_op.end())
      return true;
  }

  return false;
}

/// @brief Checks if a node is a target of constant folding transform
bool foldable(const loco::Node *node)
{
  if (node->dialect() == loco::CanonicalDialect::get())
  {
    if (skip(node))
      return false;

    if (node->arity() == 0) // e.g., when a node is e.g, ConstGen or Pull
      return false;

    // When all args are ConstGen, let's do Constant Folding Transforms
    for (int i = 0; i < node->arity(); i++)
    {
      if (node->arg(i)->opnum() != static_cast<uint32_t>(loco::CanonicalOpcode::ConstGen))
        return false;
    }

    return true;
  }
  else
  {
    return false;
  }
}

void fold(loco::Graph *graph, loco::Node *node)
{
  assert(foldable(node)); // sanity check to find a mistake when this function is reused later

  // calcluate foldable node
  locomotiv::Session sess(graph, std::vector<loco::Node *>{node});
  sess.infer();
  auto data = sess.get_output(0);

  assert(data != nullptr);

  auto shape = data->shape();
  auto dtype = data->dtype();

  // build ConstGen
  auto new_const = graph->nodes()->create<loco::ConstGen>();
  {
    new_const->dtype(dtype);

    new_const->rank(shape->rank());
    for (int d = 0; d < shape->rank(); d++)
      new_const->dim(d) = shape->dim(d);

    auto count = num_elements(*new_const);

    if (dtype == loco::DataType::FLOAT32)
    {
      new_const->size<loco::DataType::FLOAT32>(count);

      auto const_buf = data->as_f32_bufptr()->base();
      for (int x = 0; x < count; x++)
        new_const->at<loco::DataType::FLOAT32>(x) = const_buf[x];
    }
    else if (dtype == loco::DataType::S32)
    {
      new_const->size<loco::DataType::S32>(count);

      auto const_buf = data->as_s32_bufptr()->base();
      for (int x = 0; x < count; x++)
        new_const->at<loco::DataType::S32>(x) = const_buf[x];
    }
  }

  // replace node with new_const
  loco::replace(node).with(new_const);
}

} // namespace

namespace logo
{

bool ConstantFoldingPass::run(loco::Graph *graph)
{
  auto outputs = loco::output_nodes(graph);

  bool changed = false;
  for (auto node : loco::postorder_traversal(outputs))
  {
    if (foldable(node))
    {
      fold(graph, node);
      changed = true;
    }
  }

  return changed;
}

} // namespace logo
