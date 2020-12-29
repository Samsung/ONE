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

#include "GraphDumper.h"

#include "ir/Graph.h"
#include "compiler/LoweredGraph.h"
#include "util/logging.h"
#include "misc/string_helpers.h"

namespace onert
{
namespace dumper
{
namespace text
{

namespace
{

std::string formatOperandIndexSequence(const ir::OperandIndexSequence &seq)
{
  std::vector<std::string> strs;
  for (auto ind : seq)
    strs.push_back(dumper::text::formatOperandBrief(ind));
  return nnfw::misc::join(strs.begin(), strs.end(), ", ");
}

} // namespace

std::string formatOperandBrief(ir::OperandIndex ind)
{
  std::stringstream ss;
  ss << ind;
  return ss.str();
}

std::string formatOperand(const ir::Graph &, ir::OperandIndex ind)
{
  std::stringstream ss;
  ss << ind;
  // TODO Print shape, type and maybe more
  return ss.str();
}

std::string formatOperation(const ir::Graph &graph, ir::OperationIndex ind)
{
  std::stringstream ss;
  const auto &op = graph.operations().at(ind);

  ss << formatOperandIndexSequence(op.getOutputs());
  ss << " = ";
  ss << ind << "_" << op.name() << "(";
  ss << formatOperandIndexSequence(op.getInputs());
  ss << ")";
  return ss.str();
}

std::string formatOpSequence(const compiler::LoweredGraph &lgraph, ir::OpSequenceIndex ind)
{
  std::stringstream ss;
  const auto &lower_info = lgraph.getLowerInfo()->op_seq.at(ind);

  const auto &op_seq = lgraph.op_seqs().at(ind);
  std::string inputs_str = formatOperandIndexSequence(op_seq.getInputs());
  std::string outputs_str = formatOperandIndexSequence(op_seq.getOutputs());
  VERBOSE(GraphDumper) << ind << " {" << lower_info->backend()->config()->id() << "} In("
                       << inputs_str << ") Out(" << outputs_str << "):" << std::endl;
  for (auto op_ind : op_seq.operations())
  {
    ss << "  " << formatOperation(lgraph.graph(), op_ind) << "\n";
  }
  return ss.str();
}

void dumpGraph(const ir::Graph &graph)
{
  VERBOSE(GraphDumper) << "{\n";
  auto ops_topol = graph.topolSortOperations();
  for (auto op_ind : ops_topol)
  {
    VERBOSE(GraphDumper) << "  " << formatOperation(graph, op_ind) << "\n";
  }
  VERBOSE(GraphDumper) << "}\n";
  VERBOSE(GraphDumper) << std::endl;
}

void dumpLoweredGraph(const compiler::LoweredGraph &lgraph)
{
  VERBOSE(GraphDumper) << "{\n";
  // TODO Could logging system handle this? (Inserting prefix for each line)
  lgraph.iterateTopolOpSeqs([&](const ir::OpSequenceIndex &ind, const ir::OpSequence &) {
    std::istringstream iss{dumper::text::formatOpSequence(lgraph, ind)};
    std::string line;
    while (std::getline(iss, line))
      VERBOSE(GraphDumper) << line << std::endl;
  });
  VERBOSE(GraphDumper) << "}\n";
  VERBOSE(GraphDumper) << std::endl;
}

} // namespace text
} // namespace dumper
} // namespace onert
