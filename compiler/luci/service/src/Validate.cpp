/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/Validate.h"

#include <luci/IR/Nodes/CircleOutput.h>
#include <luci/Log.h>

#include <loco/IR/NodeShape.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/TypeInference.h>

#include <cassert>
#include <vector>

namespace
{

std::ostream &operator<<(std::ostream &os, const loco::TensorShape &tensor_shape)
{
  os << "[";
  for (uint32_t r = 0; r < tensor_shape.rank(); ++r)
  {
    if (r)
      os << ",";
    os << tensor_shape.dim(r).value();
  }
  os << "]";
  return os;
}

/**
 * @brief  returns a node that is CircleOutput with index is out_index in nodes
 */
luci::CircleOutput *find_node(std::vector<loco::Node *> nodes, loco::GraphOutputIndex out_index)
{
  for (auto node : nodes)
  {
    auto circle_output = dynamic_cast<luci::CircleOutput *>(node);
    if (circle_output != nullptr)
    {
      if (circle_output->indexed() && circle_output->index() == out_index)
        return circle_output;
    }
  }
  return nullptr;
}

bool validate_shape_dtype(loco::Graph *g)
{
  LOGGER(l);

  auto output_nodes = loco::output_nodes(g);

  auto count = g->outputs()->size();
  for (uint32_t out = 0; out < count; ++out)
  {
    auto graph_out = g->outputs()->at(out);
    auto out_index = graph_out->index();

    auto circle_output = find_node(output_nodes, out_index);
    assert(circle_output != nullptr);
    assert(circle_output->from() != nullptr);
    auto circle_node = loco::must_cast<luci::CircleNode *>(circle_output->from());
    assert(loco::shape_known(circle_node));

    // check if output node shape is same as graph output shape
    auto co_tensor_shape = loco::shape_get(circle_node).as<loco::TensorShape>();
    auto go_tensor_shape = graph_out->shape();
    assert(go_tensor_shape);
    if (!(co_tensor_shape == *go_tensor_shape))
    {
      INFO(l) << "[luci] Shape for output #" << out_index << " not same " << std::endl;
      INFO(l) << "[luci]    " << circle_node->name() << " " << co_tensor_shape << " vs "
              << *go_tensor_shape << std::endl;
      return false;
    }

    // check if data type match
    assert(loco::dtype_known(circle_node));
    if (graph_out->dtype() != loco::dtype_get(circle_node))
    {
      INFO(l) << "[luci] Type for output #" << out_index << " not same " << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace

namespace luci
{

bool validate(loco::Graph *g)
{
  if (!loco::valid(g))
    return false;

  if (!validate_shape_dtype(g))
    return false;

  // TODO add more validation

  return true;
}

} // namespace luci
