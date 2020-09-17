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
#include "DotSubgraphInfo.h"
#include "ir/OpSequence.h"
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

void DotDumper::dump(const std::string &tag)
{
  if (_level == Level::OFF)
  {
    return;
  }

  onert::dumper::dot::DotBuilder dot_builder;

  auto &operations = _graph.operations();
  auto &operands = _graph.operands();

  ir::OperationIndexMap<std::unique_ptr<Operation>> operation_nodes;
  std::unordered_map<ir::OperandIndex, std::unique_ptr<Operand>> operand_nodes;

  auto backend_to_fillcolor = [](const backend::Backend *backend) {
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
  };

  util::Set<ir::OperandIndex> shown_operand_set;

  operands.iterate([&](const ir::OperandIndex &index, const ir::Operand &object) {
    bool showing_cond = false;
    if (_level == Level::ALL)
    {
      showing_cond = true;
    }
    else
    {
      showing_cond =
          !object.isConstant() || (_graph.getInputs() + _graph.getOutputs()).contains(index);
    }
    if (showing_cond)
    {
      shown_operand_set.add(index);

      auto type = [&]() {
        using onert::dumper::dot::Operand;
        if (_graph.getInputs().contains(index))
          return Operand::Type::MODEL_INPUT;
        if (_graph.getOutputs().contains(index))
          return Operand::Type::MODEL_OUTPUT;
        return Operand::Type::INTERNAL;
      }();

      auto node = std::make_unique<Operand>(index, type);

      {
        // Display LowerInfo attributes
        std::string label = std::to_string(index.value());
        std::string fillcolor = "";
        if (_lowered_graph)
        {
          auto lower_info = _lowered_graph->getLowerInfo(index);
          const auto &def_factors = lower_info->def_factors();
          if (def_factors.size() > 0)
          {
            label += "\\n[";
            label += def_factors.getOnlyElement().backend()->config()->id();
            label += "]";

            fillcolor = backend_to_fillcolor(lower_info->def_factors().getOnlyElement().backend());
          }
        }
        node->setAttribute("label", label);
        node->setAttribute("fillcolor", fillcolor);
      }

      operand_nodes.emplace(index, std::move(node));
    }
  });

  operations.iterate([&](const ir::OperationIndex &index, const ir::Operation &op) {
    auto node = std::make_unique<Operation>(index, op);

    for (auto input : op.getInputs())
    {
      using onert::dumper::dot::Operand;

      // Constant input and dump level is ALL_BUT_CONSTANTS
      if (operand_nodes.find(input) == operand_nodes.end())
        continue;

      auto &input_node = operand_nodes.at(input);
      input_node->addOutEdge(node.get());
    }

    for (auto output : op.getOutputs())
    {
      using onert::dumper::dot::Operand;
      auto &output_node = operand_nodes.at(output);
      node->addOutEdge(output_node.get());
    }

    operation_nodes.emplace(index, std::move(node));
  });

  if (_lowered_graph)
  {
    const auto &op_seqs = _lowered_graph->op_seqs();
    op_seqs.iterate([&](const ir::OpSequenceIndex &index, const ir::OpSequence &op_seq) {
      const auto lower_info = _lowered_graph->getLowerInfo(index);
      auto fillcolor = backend_to_fillcolor(lower_info->backend());
      std::string label =
          std::to_string(index.value()) + " [" + lower_info->backend()->config()->id() + "]";
      DotSubgraphInfo subgraph_info{index, op_seq, shown_operand_set, _graph.operations()};
      subgraph_info.label(label);
      subgraph_info.fillcolor(fillcolor);
      dot_builder.addOpSequence(subgraph_info);

      // Set fillcolor of all operations in the op_seq
      for (const auto &op_idx : op_seq.operations())
      {
        auto found = operation_nodes.find(op_idx);
        if (found != operation_nodes.end())
        {
          auto &&op = found->second;
          op->setAttribute("fillcolor", fillcolor);
        }
      }
    });
  }

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

} // namespace dot
} // namespace dumper
} // namespace onert
