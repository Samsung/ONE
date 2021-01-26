/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SubgraphContext.h"

#include <unordered_set>

namespace
{
Halide::Type transform_type(loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      return Halide::Type(Halide::Type::Float, 32, 1);
    case loco::DataType::FLOAT64:
      return Halide::Type(Halide::Type::Float, 64, 1);
    case loco::DataType::S32:
      return Halide::Type(Halide::Type::Int, 32, 1);
    case loco::DataType::S64:
      return Halide::Type(Halide::Type::Int, 64, 1);
    default:
      assert("NYI");
  }
  return Halide::Type();
}

} // unnamed namespace

namespace luci_codegen
{

void SubgraphContext::finish_construction()
{
  std::unordered_set<luci::CircleNode *> in_graph;
  std::unordered_set<luci::CircleNode *> in_inputs;
  for (auto *node: _nodes)
  {
    in_graph.insert(node);
    _generated_funcs[node]; // create object
  }
  // gather inputs and  outputs
  for (auto *node: _nodes)
  {
    for (int i = 0; i < node->arity(); ++i)
    {
      assert(dynamic_cast<luci::CircleNode *>(node->arg(i)));
      luci::CircleNode *prev = static_cast<luci::CircleNode *>(node->arg(i));
      if (in_graph.count(prev) == 0 && in_inputs.count(prev) == 0)
      {
        auto graph_input = Halide::ImageParam(transform_type(node->dtype()), node->rank());
        _inputs.push_back({prev, graph_input});
        in_inputs.insert(prev);
      }

    }
    for (auto loco_succ: loco::succs(node))
    {
      assert(dynamic_cast<luci::CircleNode *>(loco_succ));
      luci::CircleNode *succ = static_cast<luci::CircleNode *>(loco_succ);
      if (in_graph.count(succ) == 0)
      {
        _outputs.push_back({node, _generated_funcs[node]});
        break;
      }
    }
  }
#ifndef NO_DEBUG
  _constructed = true;
#endif
}

Halide::Func SubgraphContext::get_func(luci::CircleNode *node) const
{
  assert(_constructed);
  auto in_body = _generated_funcs.find(node);
  if (in_body != _generated_funcs.end())
  {
    return in_body->second;
  }
  for (auto input: _inputs)
  {
    if (input.first == node)
      return input.second;
  }
  assert(false && "target node does not belong to subgraph");
  return Halide::Func();
}


} // namespace luci_codegen
