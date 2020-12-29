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
    os << (tensor_shape.dim(r).known() ? tensor_shape.dim(r).value() : -1);
  }
  os << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const luci::CircleNode *circle_node)
{
  os << "[";
  for (uint32_t r = 0; r < circle_node->rank(); ++r)
  {
    if (r)
      os << ",";
    os << (circle_node->dim(r).known() ? circle_node->dim(r).value() : -1);
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

    // Shape and dtype validation for CiecleOutputExclude is not needed
    if (dynamic_cast<luci::CircleOutputExclude *>(circle_node))
      continue;

    assert(circle_node->shape_status() != luci::ShapeStatus::UNDEFINED);

    // check if output node shape is same as graph output shape
    auto go_tensor_shape = graph_out->shape();
    assert(go_tensor_shape);

    bool is_shape_valid = (circle_node->rank() == go_tensor_shape->rank());
    for (uint32_t i = 0; is_shape_valid && i < circle_node->rank(); ++i)
      if (circle_node->dim(i).known() && go_tensor_shape->dim(i).known() && circle_node->dim(i).value() != go_tensor_shape->dim(i).value())
        is_shape_valid = false;

    if (is_shape_valid == false)
    {
      INFO(l) << "[luci] Shape for output #" << out_index << " not same " << std::endl;
      INFO(l) << "[luci]    " << circle_node->name() << " " << circle_node << " vs "
              << *go_tensor_shape << std::endl;
      return false;
    }

    // check if data type match
    assert(circle_node->dtype() != loco::DataType::Unknown);
    if (graph_out->dtype() != circle_node->dtype())
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
