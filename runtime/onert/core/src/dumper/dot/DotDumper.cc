/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <fstream>
#include <unordered_map>

#include "DotDumper.h"
#include "DotBuilder.h"
#include "ir/OperandIndexMap.h"
#include "ir/OperationIndexMap.h"
#include "backend/Backend.h"
#include "backend/IConfig.h"
#include "compiler/BackendManager.h"

namespace onert
{
namespace dumper
{
namespace dot
{

namespace
{
std::string backend_to_fillcolor(const backend::Backend *backend)
{
  static const auto map = []() {
    std::unordered_map<const backend::Backend *, std::string> ret;
    uint32_t index = 1; // Start from 1 to avoid 0(red) which is too dark :(
    for (const auto backend : compiler::BackendManager::get().getAll())
    {
      ret.emplace(backend, Node::BG_COLORS[index]);
      index = (index + 1) % (sizeof(Node::BG_COLORS) / sizeof(Node::BG_COLORS[0]));
    }
    return ret;
  }();
  auto itr = map.find(backend);
  if (itr == map.end())
  {
    return Node::DEFAULT_FILLCOLOR;
  }
  else
  {
    return itr->second;
  }
}

std::unordered_map<ir::OperandIndex, std::unique_ptr<Operand>>
generate_dot_operands(const ir::Graph &graph, const DotDumper::Level level)
{
  std::unordered_map<ir::OperandIndex, std::unique_ptr<Operand>> dot_operands;

  const auto &operands = graph.operands();
  operands.iterate([&](const ir::OperandIndex &index, const ir::Operand &object) {
    bool showing_cond =
      level == DotDumper::Level::ALL
        ? true
        : !object.isConstant() || (graph.getInputs() + graph.getOutputs()).contains(index);
    if (showing_cond)
    {
      auto type = [&]() {
        using onert::dumper::dot::Operand;
        if (graph.getInputs().contains(index))
          return Operand::Type::MODEL_INPUT;
        if (graph.getOutputs().contains(index))
          return Operand::Type::MODEL_OUTPUT;
        return Operand::Type::INTERNAL;
      }();

      auto node = std::make_unique<Operand>(index, type);
      std::string label = std::to_string(index.value());
      std::string fillcolor = "";
      node->setAttribute("label", label);
      node->setAttribute("fillcolor", fillcolor);

      dot_operands.emplace(index, std::move(node));
    }
  });

  return dot_operands;
}

ir::OperationIndexMap<std::unique_ptr<Operation>>
generate_dot_operations(const ir::Graph &graph,
                        const ir::OperandIndexMap<std::unique_ptr<Operand>> &dot_operands)
{
  ir::OperationIndexMap<std::unique_ptr<Operation>> dot_operations;
  const auto &operations = graph.operations();
  operations.iterate([&](const ir::OperationIndex &index, const ir::IOperation &op) {
    auto node = std::make_unique<Operation>(index, op);

    for (auto &input : op.getInputs())
    {
      using onert::dumper::dot::Operand;

      // Constant input and dump level is ALL_BUT_CONSTANTS
      if (dot_operands.find(input) == dot_operands.end())
        continue;

      auto &input_node = dot_operands.at(input);
      input_node->addOutEdge(node.get());
    }

    for (auto &output : op.getOutputs() | ir::Remove::UNDEFINED)
    {
      using onert::dumper::dot::Operand;
      auto &output_node = dot_operands.at(output);
      node->addOutEdge(output_node.get());
    }

    dot_operations.emplace(index, std::move(node));
  });

  return dot_operations;
}

void update_lower_info(const compiler::ILoweredGraph &lowered_graph,
                       ir::OperandIndexMap<std::unique_ptr<Operand>> *dot_operands)
{
  const auto &operands = lowered_graph.graph().operands();
  operands.iterate([&](const ir::OperandIndex &index, const ir::Operand &) {
    auto itr = dot_operands->find(index);
    if (itr != dot_operands->end())
    {
      auto &node = itr->second;
      // Display LowerInfo attributes
      std::string label = node->getAttribute("label");
      std::string fillcolor = node->getAttribute("fillcolor");
      auto lower_info = lowered_graph.lower_info().operand.getRawPtr(index);
      const auto &def_factors = lower_info->def_factors();
      if (def_factors.size() > 0)
      {
        label += "\\n[";
        label += def_factors.getOnlyElement().backend()->config()->id();
        label += "]";
        fillcolor = backend_to_fillcolor(lower_info->def_factors().getOnlyElement().backend());
      }
      node->setAttribute("label", label);
      node->setAttribute("fillcolor", fillcolor);
    }
  });
}

void update_lower_info(const compiler::ILoweredGraph &lowered_graph,
                       ir::OperationIndexMap<std::unique_ptr<Operation>> *dot_operations)
{
  const auto &operations = lowered_graph.graph().operations();
  operations.iterate([&](const ir::OperationIndex &index, const ir::IOperation &) {
    const auto lower_info = lowered_graph.lower_info().operation.getRawPtr(index);
    if (lower_info)
    {
      auto fillcolor = backend_to_fillcolor(lower_info->backend());
      std::string backend_label = "[" + lower_info->backend()->config()->id() + "]";
      auto itr = dot_operations->find(index);
      if (itr != dot_operations->end())
      {
        auto &node = itr->second;
        node->setAttribute("label", node->getAttribute("label") + "\n" + backend_label);
        node->setAttribute("fillcolor", fillcolor);
      }
    }
  });
}

void dump_to_file(const ir::OperandIndexMap<std::unique_ptr<Operand>> &operand_nodes,
                  const ir::OperationIndexMap<std::unique_ptr<Operation>> &operation_nodes,
                  const std::string &tag)
{
  onert::dumper::dot::DotBuilder dot_builder;
  for (const auto &e : operation_nodes)
    dot_builder.update(*e.second);
  for (const auto &e : operand_nodes)
    dot_builder.update(*e.second);

  // Dump to file
  {
    std::string file_name;
    file_name += tag;
    file_name += ".dot";
    std::filebuf fb;

    fb.open(file_name, std::ios::out);
    std::ostream os(&fb);

    dot_builder.writeDot(os);

    fb.close();
  }
}
} // namespace

void DotDumper::dump(const ir::Graph &graph, const std::string &tag)
{
  if (_level == Level::OFF)
  {
    return;
  }

  const auto dot_operands = generate_dot_operands(graph, _level);
  const auto dot_operations = generate_dot_operations(graph, dot_operands);
  dump_to_file(dot_operands, dot_operations, tag);
}

// TODO Support gradient tensors
void DotDumper::dump(const compiler::ILoweredGraph &lowered_graph, const std::string &tag)
{
  if (_level == Level::OFF)
  {
    return;
  }

  auto dot_operands = generate_dot_operands(lowered_graph.graph(), _level);
  auto dot_operations = generate_dot_operations(lowered_graph.graph(), dot_operands);
  update_lower_info(lowered_graph, &dot_operands);
  update_lower_info(lowered_graph, &dot_operations);
  dump_to_file(dot_operands, dot_operations, tag);
}

} // namespace dot
} // namespace dumper
} // namespace onert
