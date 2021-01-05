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

    if (tensor_shape.dim(r).known())
      os << tensor_shape.dim(r).value();
    else
      os << "?";
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

    if (circle_node->dim(r).known())
      os << circle_node->dim(r).value();
    else
      os << "?";
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

    // NOTE Even if shape of graph output is [] (which means "shape inference was impossible")
    //      but shape of CircleNode is not, it can be valid case because shape inference
    //      algorithm of CircleNode may be upgraded than before. The opposite is possible either.
    //      If such cases are appeared, following validation code should be fixed.
    bool is_shape_valid = (circle_node->rank() == go_tensor_shape->rank());
    for (uint32_t i = 0; is_shape_valid && i < circle_node->rank(); ++i)
    {
      if (circle_node->dim(i).value() != go_tensor_shape->dim(i).value())
      {
        if (!circle_node->dim(i).known() || !go_tensor_shape->dim(i).known())
        {
          // If at least one of two dimensions is unknown,
          // the unknown dimension can accept any value.
          INFO(l) << "Unknown dimension is matched with known dimension" << std::endl;
        }
        else
        {
          is_shape_valid = false;
        }
      }
    }

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

bool validate_shape_signature(loco::Graph *g)
{
  LOGGER(l);

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    const auto shape_signature = circle_node->shape_signature();

    if (shape_signature.rank() == 0)
      continue;

    // Rank of shape and shape signature should be same
    if (circle_node->rank() != shape_signature.rank())
    {
      INFO(l) << "[luci] Rank of shape signature for " << circle_node->name() << " do not match"
              << std::endl;
      return false;
    }

    bool has_unknown = false;

    // If shape siganture is not -1, dimension value should be same
    for (uint32_t d = 0; d < shape_signature.rank(); ++d)
    {
      if (shape_signature.dim(d) != -1 &&
          shape_signature.dim(d) != (int32_t)(circle_node->dim(d).value()))
      {
        INFO(l) << "[luci] Dimension " << d << "of shape signature for " << circle_node->name()
                << " do not match" << std::endl;
        return false;
      }

      if (shape_signature.dim(d) == -1)
        has_unknown = true;
    }

    // Shape signature should have at least one -1 value.
    if (!has_unknown)
    {
      INFO(l) << "[luci] Shape signature in " << circle_node->name()
              << " do not have unknown dimension" << std::endl;
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

  if (!validate_shape_signature(g))
    return false;

  // TODO add more validation

  return true;
}

} // namespace luci
