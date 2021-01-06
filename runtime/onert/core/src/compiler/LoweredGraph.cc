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

#include "compiler/LoweredGraph.h"

#include <assert.h>
#include <algorithm>
#include <sstream>
#include "util/logging.h"
#include "compiler/pass/ConstantInsertionPass.h"
#include "compiler/pass/ConstantLoweringPass.h"
#include "compiler/pass/PassRunner.h"
#include "compiler/pass/PermutationOperationPass.h"
#include "compiler/pass/PermutationInsertionPass.h"
#include "compiler/pass/PermutationEliminationPass.h"
#include "dumper/text/GraphDumper.h"
#include "ir/verifier/Verifier.h"
#include "backend/Backend.h"
#include "backend/IConfig.h"
#include "compiler/BackendResolver.h"
#include "compiler/ManualScheduler.h"
#include "compiler/HEScheduler.h"
#include "util/TracingCtx.h"

namespace onert
{
namespace compiler
{

LoweredGraph::LoweredGraph(const ir::Graph &graph, const CompilerOptions &options) : _graph{graph}
{
  // set tracing_ctx for copied graph
  if (options.tracing_ctx)
  {
    auto subgraph_index = options.tracing_ctx->getSubgraphIndex(&graph);
    options.tracing_ctx->setSubgraphIndex(&_graph, subgraph_index.value());
  }

  // Build backend contexts
  auto &backend_manager = BackendManager::get();
  // Create contexts for other backends
  for (auto backend_str : options.backend_list)
  {
    backend_manager.loadBackend(backend_str);
    auto backend = backend_manager.get(backend_str);

    // TODO As the default value of backend list contains "cpu", "acl_cl" and "acl_neon", and some
    // are not available on x64 or some other platforms. So this may be a workaround for x64 and
    // we should change it back(throw if backend is not loaded) later.
    if (!backend)
    {
      VERBOSE(LoweredGraph) << "Cannot load backend - " << backend_str << std::endl;
      continue;
    }
  }
  if (backend_manager.num_backends() == 0)
    throw std::runtime_error{"No available backends loaded."};

  // TODO Move "schedule" phase out of here
  // Schedule
  std::unique_ptr<BackendResolver> backend_resolver;
  auto all_backends = backend_manager.getAll();
  if (options.he_scheduler)
  {
    auto scheduler = HEScheduler(all_backends, options);
    backend_resolver = scheduler.schedule(_graph);
    _indexed_ranks = scheduler.getIndexedRanks();
  }
  else
  {
    auto scheduler = ManualScheduler(all_backends, options);
    backend_resolver = scheduler.schedule(_graph);
  }

  {
    // operand::LowerInfo holder
    ir::OperandIndexMap<std::unique_ptr<compiler::OperandLowerInfo>> operands_lower_info;

    _graph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &) {
      operands_lower_info[index] = std::make_unique<compiler::OperandLowerInfo>();
    });

    // Make op_seqs while checking whether a op can be merged into a op_seq.
    makeOperationLowerInfo(operands_lower_info, *backend_resolver);

    VERBOSE(LoweredGraph) << "dump before permutation insertion" << std::endl;
    dumper::text::dumpLoweredGraph(*this);

    // Mandatory passes
    pass::PassRunner{}
      .append(std::make_unique<pass::ConstantInsertionPass>(*this))
      .append(std::make_unique<pass::ConstantLoweringPass>(*this))
      .run();

    // Set LowerInfo for each operand from the operand::LowerInfo holder
    manipulateLowerInfo(operands_lower_info);

    dumpLowerInfo();
  }

  // Mandatory passes
  pass::PassRunner{}
    .append(std::make_unique<pass::PermutationOperationPass>(*this))
    .append(std::make_unique<pass::PermutationInsertionPass>(*this))
    .run();

  // Optimization passes
  pass::PassRunner{}.append(std::make_unique<pass::PermutationEliminationPass>(*this)).run();

  VERBOSE(LoweredGraph) << "Dump after permutation insertion" << std::endl;
  for (auto operand : _graph.getInputs())
    VERBOSE(LoweredGraph) << "Graph Input : " << operand << std::endl;
  for (auto operand : _graph.getOutputs())
    VERBOSE(LoweredGraph) << "Graph Output : " << operand << std::endl;
  dumper::text::dumpLoweredGraph(*this);

  // Graph verifications
  {
    assert(ir::verifier::InputOutputChecker().verify(_graph));
    assert(ir::verifier::DAGChecker().verify(_graph));
    assert(ir::verifier::EdgeChecker().verify(_graph));
  }
}

const compiler::OperationLowerInfo *
LoweredGraph::getLowerInfo(const ir::OperationIndex &op_ind) const
{
  auto itr = _lower_info_map.operation.find(op_ind);
  if (itr == _lower_info_map.operation.end())
    return nullptr;
  return itr->second.get();
}

void LoweredGraph::setLowerInfo(const ir::OperationIndex &op_ind,
                                std::unique_ptr<compiler::OperationLowerInfo> &&lower_info)
{
  _lower_info_map.operation[op_ind] = std::move(lower_info);
}

void LoweredGraph::removeLowerInfo(const ir::OperationIndex &op_ind)
{
  auto &op_lower_info = _lower_info_map.operation;
  assert(op_lower_info.find(op_ind) != op_lower_info.end());
  // TODO Simplify implementation using `erase` method
  for (auto it = op_lower_info.begin(); it != op_lower_info.end(); ++it)
  {
    if (it->first == op_ind)
    {
      op_lower_info.erase(it);
      break;
    }
  }
}

const compiler::OperandLowerInfo *LoweredGraph::getLowerInfo(const ir::OperandIndex &index) const
{
  auto itr = _lower_info_map.operand.find(index);
  if (itr == _lower_info_map.operand.end())
    return nullptr;
  return itr->second.get();
}

compiler::OperandLowerInfo *LoweredGraph::getLowerInfo(const ir::OperandIndex &index)
{
  auto itr = _lower_info_map.operand.find(index);
  if (itr == _lower_info_map.operand.end())
    return nullptr;
  return itr->second.get();
}

void LoweredGraph::setLowerInfo(const ir::OperandIndex &index,
                                std::unique_ptr<compiler::OperandLowerInfo> &&lower_info)
{
  _lower_info_map.operand.insert(std::make_pair(index, std::move(lower_info)));
}

void LoweredGraph::removeLowerInfo(const ir::OperandIndex &index)
{
  _lower_info_map.operand.erase(index);
}

void LoweredGraph::makeOperationLowerInfo(
  ir::OperandIndexMap<std::unique_ptr<compiler::OperandLowerInfo>> &operands_lower_info,
  const BackendResolver &backend_resolver)
{
  graph().operations().iterate([&](const ir::OperationIndex &op_ind, const ir::Operation &) {
    const ir::Operation &op = graph().operations().at(op_ind);
    auto backend = backend_resolver.getBackend(op_ind);
    auto frontend_layout = _graph.layout();

    // The layout of each backend should be set at another place
    // TODO Change setting layout of each backend at another place
    auto backend_layout = backend->config()->supportLayout(op, frontend_layout);

    for (auto operand : op.getInputs() | ir::Remove::UNDEFINED)
    {
      auto &&lower_info = operands_lower_info.at(operand);
      lower_info->addUsePermuteFactor(PermuteFactor{backend, backend_layout});
    }
    for (auto operand : op.getOutputs() | ir::Remove::UNDEFINED)
    {
      auto &&lower_info = operands_lower_info.at(operand);
      lower_info->addDefPermuteFactor(PermuteFactor{backend, backend_layout});
    }
    setLowerInfo(op_ind, std::make_unique<compiler::OperationLowerInfo>(backend, backend_layout));
  });
}

void LoweredGraph::manipulateLowerInfo(
  ir::OperandIndexMap<std::unique_ptr<compiler::OperandLowerInfo>> &operands_lower_info)
{
  const auto builtin_backend = BackendManager::get().getBuiltin();

  // TODO Rather than using NHWC Get frontend layout of this node from IR
  auto factor = PermuteFactor{builtin_backend, ir::Layout::NHWC};
  for (auto index : _graph.getInputs() | ir::Remove::UNDEFINED)
  {
    auto &&lower_info = operands_lower_info.at(index);
    assert(lower_info->def_factors().empty());
    lower_info->addDefPermuteFactor(factor);
  }
  for (auto index : _graph.getOutputs() | ir::Remove::UNDEFINED)
  {
    auto &&lower_info = operands_lower_info.at(index);
    lower_info->addUsePermuteFactor(factor);
  }
  for (auto index : _graph.getOutputs() | ir::Remove::UNDEFINED)
  {
    auto &&lower_info = operands_lower_info.at(index);
    if (lower_info->def_factors().size() == 0)
    {
      // In case of that an operand is Graph's output and not input or output of any operation
      lower_info->addDefPermuteFactor(PermuteFactor{
        builtin_backend,
        ir::Layout::NHWC // TODO Get frontend layout of this node from IR
      });
    }
  }

  // 1. Add def of variable operand
  // 2. Set LowerInfo for each operand from the OperandLowerInfo holder
  _graph.operands().iterate([&](const ir::OperandIndex &index, ir::Operand &operand) {
    // Some inputs of an operation could be non-constant, but not existed in graph inputs/outputs
    // and not undefined operand. Those inputs must have exist as a Tensor. For example,
    // UnidirectionalSequenceLSTM operation could have state inputs such as it.
    if (operand.info().isVariable())
    {
      // The variable operand with buffer is not supported yet
      assert(operand.data() == nullptr);
      assert(operand.getUses().size() == 1 && !operand.getDef().valid());
      auto &lowered_info = operands_lower_info[index];
      assert(lowered_info->def_factors().empty());
      lowered_info->addDefPermuteFactor(lowered_info->use_factors().getOnlyElement());
    }

    setLowerInfo(index, std::move(operands_lower_info[index]));
  });
}

void LoweredGraph::dumpLowerInfo()
{
  if (::onert::util::logging::ctx.enabled() == false)
    return;

  std::map<uint32_t, std::string> dumps;

  _graph.operands().iterate([&](const ir::OperandIndex &index, ir::Operand &object) {
    std::stringstream sstream;
    if (!getLowerInfo(index)->def_factors().empty() || !getLowerInfo(index)->use_factors().empty())
    {
      auto factors_to_string = [](const PermuteFactorSet &factors) {
        std::string str;
        for (auto factor : factors)
        {
          str += factor.backend()->config()->id();
          str += "(" + to_string(factor.layout()) + ")";
          str += " ";
        }
        return "{ " + str + "}";
      };

      auto operation_index_to_string = [](const ir::OperationIndexSet &operations) {
        std::string str;
        for (auto op : operations)
        {
          str += std::to_string(op.value());
          str += " ";
        }
        return "{ " + str + "}";
      };

      const auto lower_info = getLowerInfo(index);
      const auto &shape = object.shape();
      std::string def_ops =
        object.getDef().valid() ? std::to_string(object.getDef().value()) : "N/A";
      std::string use_ops = operation_index_to_string(object.getUses());
      std::string def_layouts = factors_to_string(lower_info->def_factors());
      std::string use_layouts = factors_to_string(lower_info->use_factors());
      sstream << "Operand " << index << " LowerInfo" << std::endl;
      sstream << "  - Shape           : { ";
      for (auto i = 0; i < shape.rank(); ++i)
      {
        sstream << (shape.dim(i)) << " ";
      }
      sstream << "}" << std::endl;
      sstream << "  - Def Operations  : " << def_ops << std::endl;
      sstream << "  - Use Operations  : " << use_ops << std::endl;
      sstream << "  - Data            : "
              << (object.data() ? (std::to_string(object.data()->size()) + " bytes") : "N/A")
              << std::endl;
      sstream << "  - Lower Info" << std::endl;
      sstream << "    - Def Backends    : " << def_layouts << std::endl;
      sstream << "    - Use Backends    : " << use_layouts << std::endl;
    }
    dumps.emplace(index.value(), sstream.str());
  });

  for (const auto &e : dumps)
  {
    if (!e.second.empty())
    {
      std::istringstream iss(e.second);
      std::string line;
      while (std::getline(iss, line))
        VERBOSE(Lower) << line << std::endl;
    }
  }
}

} // namespace compiler
} // namespace onert
