/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OpSelector.h"

#include <luci/ConnectNode.h>
#include <luci/Profile/CircleNodeID.h>
#include <luci/Service/CircleNodeClone.h>

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

namespace
{

/**
 * @brief Tokenize given string
 *
 * Assumes given string looks like below.
 *
 * - '1,2,5,7,9'
 * - '1-5,6,7,9,12-14'
 * - 'tensor_a,tensor_b,tensor_d'
 *
 * NOTE. 1-5 is same with '1,2,3,4,5'.
 *
 * WARNING. SelectType::NAME doesn't allow '-' like 'tensor_a-tensor_c'.
 */
std::vector<std::string> split_into_vector(const std::string &str, const char &delim)
{
  std::vector<std::string> ret;
  std::istringstream is(str);
  for (std::string item; std::getline(is, item, delim);)
  {
    ret.push_back(item);
  }

  // Remove empty string
  ret.erase(std::remove_if(ret.begin(), ret.end(), [](const std::string &s) { return s.empty(); }),
            ret.end());

  return ret;
}

bool is_number(const std::string &s)
{
  return !s.empty() && std::find_if(s.begin(), s.end(),
                                    [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

bool is_number(const std::vector<std::string> &vec)
{
  for (const auto &s : vec)
  {
    if (not::is_number(s))
    {
      return false;
    }
  }
  return true;
}

// TODO Move this class into a separate header for reuse
class IsMultiOutputNode final : public luci::CircleNodeVisitor<bool>
{
public:
  bool visit(const luci::CircleCustom *) final { return true; }
  bool visit(const luci::CircleIf *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV4 *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV5 *) final { return true; }
  bool visit(const luci::CircleSplit *) final { return true; }
  bool visit(const luci::CircleSplitV *) final { return true; }
  bool visit(const luci::CircleTopKV2 *) final { return true; }
  bool visit(const luci::CircleUnique *) final { return true; }
  bool visit(const luci::CircleUnpack *) final { return true; }
  bool visit(const luci::CircleWhile *) final { return true; }
  // default is false
  bool visit(const luci::CircleNode *) final { return false; }
};

std::unique_ptr<loco::Graph> make_graph(const std::vector<const luci::CircleNode *> nodes)
{
  auto graph = loco::make_graph();

  luci::CloneContext ctx;
  // clone nodes
  for (const auto &n : nodes)
  {
    auto clone = luci::clone_node(n, graph.get());
    ctx.emplace(n, clone);
  }
  // set graph input
  for (const auto &n : nodes)
  {
    for (uint32_t i = 0; i < n->arity(); i++)
    {
      auto arg = n->arg(i);
      auto input_node = dynamic_cast<luci::CircleNode *>(arg);
      auto ctx_it = ctx.find(input_node);
      // check if the node already has been cloned
      if (ctx_it != ctx.end())
        continue;
      // the node isn't graph input if it is an other node's input
      if (std::find(nodes.begin(), nodes.end(), arg) != nodes.end())
        continue;
      // do not set graph input if the node is CircleOutputExclude.
      auto circle_output_exclude = dynamic_cast<luci::CircleOutputExclude *>(arg);
      if (circle_output_exclude)
      {
        auto clone = luci::clone_node(circle_output_exclude, graph.get());
        ctx.emplace(circle_output_exclude, clone);
        continue;
      }
      auto circle_const = dynamic_cast<luci::CircleConst *>(arg);
      if (circle_const != nullptr)
      {
        auto clone = luci::clone_node(circle_const, graph.get());
        ctx.emplace(circle_const, clone);
      }
      else
      {
        // circle input
        auto circle_input = graph->nodes()->create<luci::CircleInput>();
        input_node = dynamic_cast<luci::CircleNode *>(arg);
        if (not input_node)
        {
          throw std::runtime_error{"ERROR: Invalid graph"};
        }
        luci::copy_common_attributes(input_node, circle_input);
        ctx.emplace(input_node, circle_input);
        // graph input
        auto graph_input = graph->inputs()->create();
        graph_input->name(circle_input->name());
        graph_input->dtype(circle_input->dtype());
        // graph input shape
        auto input_shape = std::make_unique<loco::TensorShape>();
        input_shape->rank(circle_input->rank());
        for (uint32_t i = 0; i < circle_input->rank(); i++)
        {
          if (circle_input->dim(i).known())
          {
            circle_input->dim(i).set(circle_input->dim(i).value());
          }
        }
        graph_input->shape(std::move(input_shape));

        circle_input->index(graph_input->index());
      }
    }
  }

  const auto original_graph = nodes.at(0)->graph();
  const auto original_outputs = loco::output_nodes(const_cast<loco::Graph *>(original_graph));

  // set graph output
  for (auto &n : nodes)
  {
    auto outputs = loco::succs(n);
    bool beingUsed = false;
    for (const auto &o : outputs)
    {
      if (std::find(nodes.begin(), nodes.end(), o) != nodes.end())
      {
        beingUsed = true;
        break;
      }
    }

    bool originalOutput = false;
    for (const auto &o : outputs)
    {
      if (std::find(original_outputs.begin(), original_outputs.end(), o) != original_outputs.end())
      {
        originalOutput = true;
        break;
      }
    }

    // the node isn't graph output if it is an other node's output
    if (beingUsed and not originalOutput)
      continue;

    IsMultiOutputNode multiout_visitor;
    bool isMultiOut = n->accept(&multiout_visitor);
    for (auto &o : outputs)
    {
      const luci::CircleNode *output_node = nullptr;
      if (isMultiOut)
      {
        output_node = dynamic_cast<const luci::CircleNode *>(o);
        if (not output_node)
        {
          throw std::runtime_error{"ERROR: Invalid graph"};
        }
      }
      else
      {
        output_node = n;
      }
      // circle output
      auto circle_output = graph->nodes()->create<luci::CircleOutput>();
      luci::copy_common_attributes(output_node, circle_output);
      // connect to cloned output node
      circle_output->from(ctx.find(output_node)->second);
      // graph output
      auto graph_output = graph->outputs()->create();
      graph_output->name(output_node->name());
      graph_output->dtype(output_node->dtype());
      // graph output shape
      auto output_shape = std::make_unique<loco::TensorShape>();
      output_shape->rank(circle_output->rank());
      for (uint32_t i = 0; i < output_shape->rank(); i++)
      {
        if (circle_output->dim(i).known())
        {
          output_shape->dim(i).set(circle_output->dim(i).value());
        }
      }
      graph_output->shape(std::move(output_shape));

      circle_output->index(graph_output->index());
      if (not isMultiOut)
        break;
    }
  }
  // connect nodes
  for (const auto &n : nodes)
  {
    luci::clone_connect(n, ctx);
  }

  return graph;
}

} // namespace

namespace opselector
{

OpSelector::OpSelector(const luci::Module *module) : _module{module}
{
  if (_module->size() != 1)
  {
    throw std::runtime_error{"ERROR: Not support two or more subgraphs"};
  }
}

template <>
std::vector<const luci::CircleNode *>
OpSelector::select_by<SelectType::ID>(const std::vector<std::string> &comma_tokens)
{
  std::vector<uint32_t> by_id;

  for (const auto &comma_token : comma_tokens)
  {
    auto dash_tokens = ::split_into_vector(comma_token, '-');
    if (not::is_number(dash_tokens))
    {
      throw std::runtime_error{
        "ERROR: To select operator by id, please use these args: [0-9], '-', ','"};
    }

    // Convert string into integer
    std::vector<uint32_t> int_tokens;
    try
    {
      std::transform(dash_tokens.begin(), dash_tokens.end(), std::back_inserter(int_tokens),
                     [](const std::string &str) { return static_cast<uint32_t>(std::stoi(str)); });
    }
    catch (const std::out_of_range &)
    {
      // Uf input is big integer like '123467891234', stoi throws this exception.
      throw std::runtime_error{"ERROR: Argument is out of range."};
    }
    catch (...)
    {
      throw std::runtime_error{"ERROR: Unknown error"};
    }

    switch (int_tokens.size())
    {
      case 0: // inputs like "-"
      {
        throw std::runtime_error{"ERROR: Nothing was entered"};
      }
      case 1: // inputs like "1", "2"
      {
        by_id.push_back(int_tokens.at(0));
        break;
      }
      case 2: // inputs like "1-2", "11-50"
      {
        for (uint32_t i = int_tokens.at(0); i <= int_tokens.at(1); i++)
        {
          by_id.push_back(i);
        }
        break;
      }
      default: // inputs like "1-2-3"
      {
        throw std::runtime_error{"ERROR: Too many '-' in str."};
      }
    }
  }

  loco::Graph *graph = _module->graph(0);
  std::vector<const luci::CircleNode *> selected_nodes;

  for (auto node : loco::all_nodes(graph))
  {
    auto cnode = loco::must_cast<const luci::CircleNode *>(node);

    try
    {
      auto node_id = luci::get_node_id(cnode);
      for (auto selected_id : by_id)
      {
        if (selected_id == node_id)
        {
          selected_nodes.emplace_back(cnode);
        }
      }
    }
    catch (const std::runtime_error &)
    {
      continue;
    }
  }

  return selected_nodes;
}

template <>
std::vector<const luci::CircleNode *>
OpSelector::select_by<SelectType::NAME>(const std::vector<std::string> &tokens)
{
  loco::Graph *graph = _module->graph(0);
  std::vector<const luci::CircleNode *> selected_nodes;

  for (auto node : loco::all_nodes(graph))
  {
    auto cnode = loco::must_cast<const luci::CircleNode *>(node);
    std::string node_name = cnode->name();

    for (const auto &selected_name : tokens)
      if (selected_name.compare(node_name) == 0) // find the selected name
        selected_nodes.emplace_back(cnode);
  }

  return selected_nodes;
}

template <SelectType SELECT_TYPE>
std::unique_ptr<luci::Module> OpSelector::select_by(const std::string &str)
{
  auto colon_tokens = ::split_into_vector(str, ',');
  if (colon_tokens.empty())
  {
    throw std::runtime_error{"ERROR: Nothing was entered."};
  }

  assert(_module->size() == 1);

  auto selected_nodes = select_by<SELECT_TYPE>(colon_tokens);

  // multiout node should be considered
  IsMultiOutputNode multiout_visitor;
  std::vector<const luci::CircleNode *> output_nodes;
  for (const auto &node : selected_nodes)
  {
    if (node->accept(&multiout_visitor))
    {
      auto outputs = loco::succs(node);
      for (auto &o : outputs)
      {
        output_nodes.push_back(dynamic_cast<luci::CircleNode *>(o));
      }
    }
  }
  selected_nodes.insert(selected_nodes.end(), output_nodes.begin(), output_nodes.end());

  auto new_module = std::make_unique<luci::Module>();
  new_module->add(::make_graph(selected_nodes));

  return new_module;
}

template std::unique_ptr<luci::Module>
OpSelector::select_by<SelectType::ID>(const std::string &str);

template std::unique_ptr<luci::Module>
OpSelector::select_by<SelectType::NAME>(const std::string &str);

} // namespace opselector
