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

#include "loco/IR/Graph.h"

#include <stdex/Memory.h>

#include <cassert>

namespace
{

std::unique_ptr<loco::TensorShape> make_tensor_shape(std::initializer_list<loco::Dimension> dims)
{
  auto tensor_shape = stdex::make_unique<loco::TensorShape>();

  tensor_shape->rank(dims.size());
  {
    uint32_t axis = 0;
    for (auto it = dims.begin(); it != dims.end(); ++it)
    {
      tensor_shape->dim(axis++) = *it;
    }
    assert(axis == dims.size());
  }

  return tensor_shape;
}

} // namespace

namespace loco
{

void Mixin<Trait::TensorShaped>::shape(std::initializer_list<Dimension> dims)
{
  shape(make_tensor_shape(dims));
}

GraphInput *Graph::InputContext::create(void)
{
  return take(stdex::make_unique<GraphInput>(size()));
}

GraphOutput *Graph::OutputContext::create(void)
{
  return take(stdex::make_unique<GraphOutput>(size()));
}

std::set<loco::Node *> all_nodes(loco::Graph *g)
{
  std::set<loco::Node *> res;

  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    res.insert(g->nodes()->at(n));
  }

  return res;
}

std::vector<Node *> input_nodes(const Graph *g)
{
  std::map<GraphInputIndex, loco::Node *> table;

  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    auto node = g->nodes()->at(n);

    if (auto service = node->dialect()->service<GraphInputIndexQueryService>())
    {
      if (service->associated(node))
      {
        auto input_index = service->index(node);
        assert(table.find(input_index) == table.end());
        table[input_index] = node;
      }
    }
  }

  std::vector<loco::Node *> res;

  for (uint32_t n = 0; n < g->inputs()->size(); ++n)
  {
    auto it = table.find(n);
    res.emplace_back(it == table.end() ? nullptr : it->second);
  }

  return res;
}

std::vector<loco::Node *> output_nodes(loco::Graph *g)
{
  std::map<GraphOutputIndex, loco::Node *> table;

  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    auto node = g->nodes()->at(n);

    if (auto service = node->dialect()->service<GraphOutputIndexQueryService>())
    {
      if (service->associated(node))
      {
        auto output_index = service->index(node);
        assert(table.find(output_index) == table.end());
        table[output_index] = node;
      }
    }
  }

  std::vector<loco::Node *> res;

  for (uint32_t n = 0; n < g->outputs()->size(); ++n)
  {
    auto it = table.find(n);
    res.emplace_back(it == table.end() ? nullptr : it->second);
  }

  return res;
}

std::unique_ptr<Graph> make_graph(void) { return std::unique_ptr<Graph>{new Graph}; }

} // namespace loco
