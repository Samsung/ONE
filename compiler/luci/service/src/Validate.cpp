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
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>
#include <luci/LogHelper.h>

#include <loco/IR/NodeShape.h>

#include <cassert>
#include <unordered_map>
#include <vector>
#include <iostream>

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

// TODO Reduce duplicate with validate_shape_dtype
bool validate_shape(loco::Graph *g)
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

    // Shape validation for CircleOutputExclude is not needed
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
      if (!circle_node->dim(i).known() || !go_tensor_shape->dim(i).known())
      {
        // If at least one of two dimensions is unknown,
        // the unknown dimension can accept any value.
        INFO(l) << "Unknown dimension is matched with known dimension" << std::endl;
      }
      else if (circle_node->dim(i).value() != go_tensor_shape->dim(i).value())
      {
        is_shape_valid = false;
      }
    }

    if (is_shape_valid == false)
    {
      INFO(l) << "[luci] Shape for output #" << out_index << " not same " << std::endl;
      INFO(l) << "[luci]    " << circle_node->name() << " " << circle_node << " vs "
              << *go_tensor_shape << std::endl;
      return false;
    }
  }

  return true;
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

    // Shape and dtype validation for CircleOutputExclude is not needed
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
      if (!circle_node->dim(i).known() || !go_tensor_shape->dim(i).known())
      {
        // If at least one of two dimensions is unknown,
        // the unknown dimension can accept any value.
        INFO(l) << "Unknown dimension is matched with known dimension" << std::endl;
      }
      else if (circle_node->dim(i).value() != go_tensor_shape->dim(i).value())
      {
        is_shape_valid = false;
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

class MultiOutNodeValidate final : public luci::CircleNodeVisitor<bool>
{
public:
  MultiOutNodeValidate() {}

private:
  template <class T> bool check(const luci::CircleNode *node)
  {
    auto succs = loco::succs(node);
    if (succs.size() < 1)
      return false;
    for (const auto &cnode : succs)
    {
      auto const child = dynamic_cast<const T *>(cnode);
      if (child == nullptr)
        return false;
    }
    return true;
  }

public:
  bool visit(const luci::CircleBidirectionalSequenceLSTM *node) final
  {
    return check<luci::CircleBidirectionalSequenceLSTMOut>(node);
  }
  bool visit(const luci::CircleCustom *node) final { return check<luci::CircleCustomOut>(node); }
  bool visit(const luci::CircleIf *node) final { return check<luci::CircleIfOut>(node); }
  bool visit(const luci::CircleNonMaxSuppressionV4 *node) final
  {
    return check<luci::CircleNonMaxSuppressionV4Out>(node);
  }
  bool visit(const luci::CircleNonMaxSuppressionV5 *node) final
  {
    return check<luci::CircleNonMaxSuppressionV5Out>(node);
  }
  bool visit(const luci::CircleSplit *node) final { return check<luci::CircleSplitOut>(node); }
  bool visit(const luci::CircleSplitV *node) final { return check<luci::CircleSplitVOut>(node); }
  bool visit(const luci::CircleTopKV2 *node) final { return check<luci::CircleTopKV2Out>(node); }
  bool visit(const luci::CircleUnique *node) final { return check<luci::CircleUniqueOut>(node); }
  bool visit(const luci::CircleUnpack *node) final { return check<luci::CircleUnpackOut>(node); }
  bool visit(const luci::CircleWhile *node) final { return check<luci::CircleWhileOut>(node); }

  // default true for other nodes
  bool visit(const luci::CircleNode *) final { return true; }
};

/**
 * @brief Validate sequence of multi-output nodes are followed for specific
 *        IRs such as CircleIfOut.
 */
bool validate_multi_outs(loco::Graph *g)
{
  LOGGER(l);

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto const cnode = loco::must_cast<luci::CircleNode *>(node);

    MultiOutNodeValidate d;
    if (cnode->accept(&d))
      continue;

    auto const name = cnode->name();
    INFO(l) << "Node: " << name << ", " << (uint32_t)(cnode->opcode()) << " has invalid successor."
            << std::endl;

    return false;
  }

  return true;
}

class VirtualNodeDetector final : public luci::CircleNodeVisitor<bool>
{
public:
  VirtualNodeDetector() {}

public:
  bool visit(const luci::CircleBidirectionalSequenceLSTMOut *) final { return true; }
  bool visit(const luci::CircleCustomOut *) final { return true; }
  bool visit(const luci::CircleIfOut *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV4Out *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV5Out *) final { return true; }
  bool visit(const luci::CircleSplitOut *) final { return true; }
  bool visit(const luci::CircleSplitVOut *) final { return true; }
  bool visit(const luci::CircleTopKV2Out *) final { return true; }
  bool visit(const luci::CircleUnpackOut *) final { return true; }
  bool visit(const luci::CircleUniqueOut *) final { return true; }
  bool visit(const luci::CircleWhileOut *) final { return true; }
  bool visit(const luci::CircleOutputDummy *) final { return true; }
  bool visit(const luci::CircleOutputExclude *) final { return true; }

  // Return false by default
  bool visit(const luci::CircleNode *) final { return false; }
};

} // namespace

namespace luci
{

bool validate_shape(loco::Graph *g)
{
  if (!loco::valid(g))
    return false;

  if (!::validate_shape(g))
    return false;

  return true;
}

bool validate(loco::Graph *g)
{
  if (!loco::valid(g))
    return false;

  if (!validate_shape_dtype(g))
    return false;

  if (!validate_multi_outs(g))
    return false;

  // TODO add more validation

  return true;
}

bool validate_name(loco::Graph *g)
{
  auto nodes = g->nodes();
  for (uint32_t n = 0; n < nodes->size(); ++n)
  {
    auto node = loco::must_cast<luci::CircleNode *>(nodes->at(n));
    // skip virtual nodes
    VirtualNodeDetector d;
    if (node->accept(&d))
      continue;

    auto name = node->name();
    if (name.empty())
      return false;
  }

  return true;
}

bool validate_unique_name(luci::Module *m)
{
  LOGGER(l);

  std::unordered_map<std::string, bool> names_col;

  for (size_t g = 0; g < m->size(); ++g)
  {
    auto graph = m->graph(g);
    auto nodes = graph->nodes();
    for (uint32_t n = 0; n < nodes->size(); ++n)
    {
      auto node = loco::must_cast<luci::CircleNode *>(nodes->at(n));
      // skip CircleOutput as it may have same name with from() node
      auto output = dynamic_cast<luci::CircleOutput *>(node);
      if (output != nullptr)
        continue;
      // skip virtual nodes
      VirtualNodeDetector d;
      if (node->accept(&d))
        continue;

      auto name = node->name();
      INFO(l) << "Node: " << name << ", " << (uint32_t)(node->opcode()) << std::endl;
      auto it = names_col.find(name);
      if (it != names_col.end())
      {
        INFO(l) << "validate_unique_name: found duplicate " << name << ", " << graph->name()
                << std::endl;
        return false;
      }

      names_col[name] = true;
    }
    // There can exist same tensor name between different subgraphs.
    names_col.clear();
  }

  return true;
}

bool validate(luci::Module *module)
{
  LOGGER(l);

  INFO(l) << "--- validate Module -----------------------------------";

  for (size_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);

    INFO(l) << luci::fmt(graph) << std::endl;

    if (!validate(graph))
    {
      std::cerr << "ERROR: Invalid circle model" << std::endl;
      return false;
    }
    if (!validate_name(graph))
    {
      std::cerr << "ERROR: circle model has empty name" << std::endl;
      return false;
    }
  }

  if (!validate_unique_name(module))
  {
    std::cerr << "ERROR: circle model has duplicate names" << std::endl;
    return false;
  }

  return true;
}

bool validate_shape(luci::Module *module)
{
  LOGGER(l);

  INFO(l) << "--- validate shape of Module -----------------------------------";

  for (size_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);

    INFO(l) << luci::fmt(graph) << std::endl;

    if (!validate_shape(graph))
    {
      std::cerr << "ERROR: Invalid circle model" << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace luci
