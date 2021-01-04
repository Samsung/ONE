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

  void visit(luci::CircleWhile *node) final
  {
    LOGGER(l);
    INFO(l) << "CircleWhile " << node->name() << std::endl;

    auto cond_branch = node->cond_branch();
    auto body_branch = node->body_branch();
    auto num_graphs = static_cast<int32_t>(_module->size());
    (void)num_graphs;

    assert(num_graphs > 0);
    assert(cond_branch >= 0 && cond_branch < num_graphs);
    assert(body_branch >= 0 && body_branch < num_graphs);

    auto cond_graph = _module->graph(cond_branch);
    auto body_graph = _module->graph(body_branch);
    assert(cond_graph != nullptr);
    assert(body_graph != nullptr);

    node->cond_graph(cond_graph);
    node->body_graph(body_graph);
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
        auto cn = loco::must_cast<luci::CircleNode *>(node);

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
 * @brief  ValidateNodeProp will validate inter graph connections for each Nodes.
 * @note   In here, only loco::GraphInput and loco::GraphOutput are validated,
 *         since this class is for checking inter graph connections.
 *         CircleNodes such as CircleInput and CircleOutput will be validated at later steps.
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
      auto then_out = loco::must_cast<luci::CircleOutput *>(then_outputs.at(idx));
      auto else_out = loco::must_cast<luci::CircleOutput *>(else_outputs.at(idx));

      auto then_graph_output = then_graph_outputs->at(then_out->index());
      auto else_graph_output = else_graph_outputs->at(else_out->index());
      if (then_graph_output->shape()->rank() != else_graph_output->shape()->rank())
      {
        INTERNAL_EXN_V("CircleIf THEN and ELSE Graph Output rank mismatch ", idx);
      }
      for (uint32_t i = 0; i < then_graph_output->shape()->rank(); ++i)
      {
        if (then_graph_output->shape()->dim(i).known() &&
            else_graph_output->shape()->dim(i).known() &&
            then_graph_output->shape()->dim(i).value() !=
              else_graph_output->shape()->dim(i).value())
        {
          INTERNAL_EXN_V("CircleIf THEN and ELSE Graph Output dimension mismatch ", idx);
        }
      }
      if (then_graph_output->dtype() != else_graph_output->dtype())
      {
        INTERNAL_EXN_V("CircleIf THEN and ELSE Graph Output type mismatch ", idx);
      }
    }
  }

  /**
   * @note  Validate CircleWhile node 'cond' and 'body' graph input/output count
   *        shape and type
   */
  void visit(luci::CircleWhile *node) final
  {
    LOGGER(l);
    INFO(l) << "CircleWhile " << node->name() << std::endl;

    auto cond_graph = node->cond_graph();
    auto body_graph = node->body_graph();
    assert(cond_graph != nullptr);
    assert(body_graph != nullptr);

    // Check input of "cond" and input/output of "body" subgraph have the same size
    auto cond_inputs = loco::input_nodes(cond_graph);
    auto cond_outputs = loco::output_nodes(cond_graph);
    auto body_inputs = loco::input_nodes(body_graph);
    auto body_outputs = loco::output_nodes(body_graph);
    if (cond_inputs.size() != body_outputs.size())
    {
      INTERNAL_EXN("CircleWhile COND input and BODY output have different sizes");
    }
    if (cond_inputs.size() != body_inputs.size())
    {
      INTERNAL_EXN("CircleWhile COND input and BODY input have different sizes");
    }
    if (cond_outputs.size() != 1)
    {
      INTERNAL_EXN("CircleWhile COND output must have size 1");
    }
    auto cond_out = loco::must_cast<luci::CircleOutput *>(cond_outputs.at(0));
    if (cond_out->dtype() != loco::DataType::BOOL)
    {
      INTERNAL_EXN("CircleWhile COND output must have bool type");
    }

    // input of "cond" and input/output of "body" subgraph must have the same shape and type
    // First we compare input of "cond" with input of "body"
    auto cond_graph_inputs = cond_graph->inputs();
    auto body_graph_inputs = body_graph->inputs();
    for (size_t idx = 0; idx < cond_inputs.size(); ++idx)
    {
      auto cond_in = loco::must_cast<luci::CircleInput *>(cond_inputs.at(idx));
      auto body_in = loco::must_cast<luci::CircleInput *>(body_inputs.at(idx));

      auto cond_graph_input = cond_graph_inputs->at(cond_in->index());
      auto body_graph_input = body_graph_inputs->at(body_in->index());
      if (cond_graph_input->shape()->rank() != body_graph_input->shape()->rank())
      {
        INTERNAL_EXN_V("CircleWhile COND input and BODY input rank mismatch ", idx);
      }
      for (uint32_t i = 0; i < cond_graph_input->shape()->rank(); ++i)
      {
        if (cond_graph_input->shape()->dim(i).known() &&
            body_graph_input->shape()->dim(i).known() &&
            cond_graph_input->shape()->dim(i).value() != body_graph_input->shape()->dim(i).value())
        {
          INTERNAL_EXN_V("CircleWhile COND input and BODY input dimension mismatch ", idx);
        }
      }
      if (cond_graph_input->dtype() != body_graph_input->dtype())
      {
        INTERNAL_EXN_V("CircleWhile COND input and BODY input type mismatch ", idx);
      }
    }

    // Next we compare input of "cond" with output of "body"
    auto body_graph_outputs = body_graph->outputs();
    for (size_t idx = 0; idx < cond_inputs.size(); ++idx)
    {
      auto cond_in = loco::must_cast<luci::CircleInput *>(cond_inputs.at(idx));
      auto body_out = loco::must_cast<luci::CircleOutput *>(body_outputs.at(idx));

      auto cond_graph_input = cond_graph_inputs->at(cond_in->index());
      auto body_graph_output = body_graph_outputs->at(body_out->index());
      if ((cond_in->rank() != body_out->rank()))
      {
        INTERNAL_EXN_V("CircleWhile COND input and BODY output shape mismatch ", idx);
      }
      if (cond_in->rank() > 0 && body_out->rank() > 0)
      {
        if (!(*cond_graph_input->shape() == *body_graph_output->shape()))
        {
          INTERNAL_EXN_V("CircleWhile COND input and BODY output shape mismatch ", idx);
        }
      }
      if (cond_in->dtype() != body_out->dtype())
      {
        INTERNAL_EXN_V("CircleWhile COND input and BODY output type mismatch ", idx);
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
        auto cn = loco::must_cast<luci::CircleNode *>(node);

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
