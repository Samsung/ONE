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

#include "PostImport.h"

#include "luci/Import/CircleReader.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <loco.h>
#include <oops/InternalExn.h>

namespace
{

/**
 * @brief  FixInterGraphNodes will fix inter graph connections for each Nodes
 */
class FixInterGraphNodes final : public luci::CircleNodeMutableVisitor<void>
{
public:
  FixInterGraphNodes(const luci::Module *m, const luci::CircleReader &r) : _module(m), _reader(r) {}

  /**
   * @note  This will set Graph* to every CircleIf nodes 'else' and 'then'
   */
  void visit(luci::CircleIf *node) final
  {
    LOGGER(l);
    INFO(l) << "CircleIf " << node->name() << std::endl;

    auto then_branch = node->then_branch();
    auto else_branch = node->else_branch();
    auto num_graphs = static_cast<int32_t>(_module->size());
    (void)num_graphs;

    assert(num_graphs > 0);
    assert(then_branch >= 0 && then_branch < num_graphs);
    assert(else_branch >= 0 && else_branch < num_graphs);

    auto then_graph = _module->graph(then_branch);
    auto else_graph = _module->graph(else_branch);
    assert(then_graph != nullptr);
    assert(else_graph != nullptr);

    node->then_graph(then_graph);
    node->else_graph(else_graph);
  }

  void visit(luci::CircleNode *) final
  {
    // DO NOTHING
  }

private:
  const luci::Module *_module;
  const luci::CircleReader &_reader;
};

/**
 * @brief  FixInterGraph will fix inter graph connections
 */
class FixInterGraph final
{
public:
  void run(loco::Graph *g, const luci::Module *m, const luci::CircleReader &r)
  {
    for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
    {
      if (recognize(node->dialect()))
      {
        auto cn = dynamic_cast<luci::CircleNode *>(node);
        assert(cn != nullptr);

        fix(cn, m, r);
      }
    }
  }

private:
  bool recognize(const loco::Dialect *dialect) { return (dialect == luci::CircleDialect::get()); }

  void fix(luci::CircleNode *node, const luci::Module *module, const luci::CircleReader &reader)
  {
    FixInterGraphNodes fix(module, reader);
    node->accept(&fix);
  }
};

} // namespace

namespace
{
/**
 * @brief  ValidateNodeProp will validate inter graph connections for each Nodes
 */
class ValidateNodeProp final : public luci::CircleNodeMutableVisitor<void>
{
public:
  ValidateNodeProp(const luci::Module *m, const luci::CircleReader &r) : _module(m), _reader(r) {}

  /**
   * @note  Validate CircleIf node 'else' and 'then' graph input/output count
   *        shape and type
   */
  void visit(luci::CircleIf *node) final
  {
    LOGGER(l);
    INFO(l) << "CircleIf " << node->name() << std::endl;

    auto then_graph = node->then_graph();
    auto else_graph = node->else_graph();
    assert(then_graph != nullptr);
    assert(else_graph != nullptr);

    // TODO support for differnt shape; but how?
    // NODE Shape/Type inference assume below conditions

    // Check both "then" and "else" subgraph outputs are same in count
    auto then_outputs = loco::output_nodes(then_graph); // CircleOutput nodes
    auto else_outputs = loco::output_nodes(else_graph);
    if (then_outputs.size() != else_outputs.size())
    {
      INTERNAL_EXN("CircleIf THEN and ELSE Graph are not same in size");
    }

    // check outputs have same shape and dtype
    auto then_graph_outputs = then_graph->outputs(); // loco::GraphOutput items
    auto else_graph_outputs = else_graph->outputs();
    for (size_t idx = 0; idx < then_outputs.size(); ++idx)
    {
      auto then_out = dynamic_cast<luci::CircleOutput *>(then_outputs.at(idx));
      auto else_out = dynamic_cast<luci::CircleOutput *>(else_outputs.at(idx));

      auto then_graph_output = then_graph_outputs->at(then_out->index());
      auto else_graph_output = else_graph_outputs->at(else_out->index());
      if (!(*then_graph_output->shape() == *else_graph_output->shape()))
      {
        INTERNAL_EXN_V("CircleIf THEN and ELSE Graph Output shape mismatch ", idx);
      }
      if (then_graph_output->dtype() != else_graph_output->dtype())
      {
        INTERNAL_EXN_V("CircleIf THEN and ELSE Graph Output type mismatch ", idx);
      }
    }
  }

  void visit(luci::CircleNode *) final
  {
    // DO NOTHING
  }

private:
  const luci::Module *_module;
  const luci::CircleReader &_reader;
};

/**
 * @brief  ValidateGraphProp will validate inter graph node properties
 */
class ValidateGraphProp final
{
public:
  void run(loco::Graph *g, const luci::Module *m, const luci::CircleReader &r)
  {
    for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
    {
      if (recognize(node->dialect()))
      {
        auto cn = dynamic_cast<luci::CircleNode *>(node);
        assert(cn != nullptr);

        eval(cn, m, r);
      }
    }
  }

private:
  bool recognize(const loco::Dialect *dialect) { return (dialect == luci::CircleDialect::get()); }

  void eval(luci::CircleNode *node, const luci::Module *module, const luci::CircleReader &reader)
  {
    ValidateNodeProp val(module, reader);
    node->accept(&val);
  }
};

} // namespace

namespace luci
{

/**
 * @brief  Do post import actions
 */
void post_import_graph(luci::Module *module, const luci::CircleReader &reader)
{
  LOGGER(l);

  auto count = module->size();

  for (size_t s = 0; s < count; ++s)
  {
    auto g = module->graph(s);
    assert(g != nullptr);

    INFO(l) << "--- FixInterGraph " << g->name() << "-------------------------";
    FixInterGraph fix;
    fix.run(g, module, reader);
  }

  for (size_t s = 0; s < count; ++s)
  {
    auto g = module->graph(s);
    assert(g != nullptr);

    INFO(l) << "--- ValidateGraphProp " << g->name() << "---------------------";
    ValidateGraphProp prop;
    prop.run(g, module, reader);
  }

  INFO(l) << "--- post_import_graph done -------------------------------------";
}

} // namespace luci
