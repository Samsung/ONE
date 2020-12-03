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
#include <sstream>
#include "util/logging.h"
#include "compiler/pass/ConstantInsertionPass.h"
#include "compiler/pass/ConstantLoweringPass.h"
#include "compiler/pass/PassRunner.h"
#include "compiler/pass/PermutationOperationPass.h"
#include "compiler/pass/PermutationInsertionPass.h"
#include "compiler/pass/PermutationEliminationPass.h"
#include "ir/GraphIterator.h"
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

  bool linear_executor = (options.executor == "Linear");

  // Build backend contexts
  auto &backend_manager = BackendManager::get();

  // Always create Controlflow backend context
  auto cf_backend = backend_manager.getControlflow();
  _backend_contexts.emplace(
    cf_backend, cf_backend->newContext(_graph, _graph.getKernelBuilder(), linear_executor));

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

    _backend_contexts.emplace(
      backend, backend->newContext(_graph, _graph.getKernelBuilder(), linear_executor));
  }
  if (backend_manager.num_backends() == 0)
    throw std::runtime_error{"No available backends loaded."};

  // TODO Move "schedule" phase out of here
  // Schedule
  std::unique_ptr<BackendResolver> backend_resolver;
  if (options.he_scheduler)
  {
    auto scheduler = HEScheduler(_backend_contexts, options);
    backend_resolver = scheduler.schedule(_graph);
    _indexed_ranks = scheduler.getIndexedRanks();
  }
  else
  {
    auto scheduler = ManualScheduler(_backend_contexts, options);
    backend_resolver = scheduler.schedule(_graph);
  }

  {
    // operand::LowerInfo holder
    ir::OperandIndexMap<std::unique_ptr<ir::operand::LowerInfo>> operands_lower_info;

    _graph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &) {
      operands_lower_info[index] = std::make_unique<ir::operand::LowerInfo>();
    });

    // Make op_seqs while checking whether a node can be merged into a op_seq.
    makeOpSequences(operands_lower_info, options, *backend_resolver);

    _op_seqs.iterate([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
      assert(op_seq.operations().size() > 0);
      std::reverse(std::begin(op_seq.operations()), std::end(op_seq.operations()));
    });

    VERBOSE(OpSequences) << "dump before permutation insertion" << std::endl;
    dumpOpSequences(_op_seqs, _graph.operations());

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
  dumpOpSequences(_op_seqs, _graph.operations());

  // Graph verifications
  {
    assert(ir::verifier::InputOutputChecker().verify(_graph));
    assert(ir::verifier::DAGChecker().verify(_graph));
    assert(ir::verifier::EdgeConsistencyChecker().verify(_graph));
  }
}

const ir::operation::LowerInfo *
LoweredGraph::getLowerInfo(const ir::OpSequenceIndex &op_seq_index) const
{
  auto itr = _lower_info_map.op_seq.find(op_seq_index);
  if (itr == _lower_info_map.op_seq.end())
    return nullptr;
  return itr->second.get();
}

void LoweredGraph::setLowerInfo(const ir::OpSequenceIndex &op_seq_index,
                                std::unique_ptr<ir::operation::LowerInfo> &&lower_info)
{
  _lower_info_map.op_seq.insert(std::make_pair(op_seq_index, std::move(lower_info)));
}

void LoweredGraph::removeLowerInfo(const ir::OpSequenceIndex &op_seq_index)
{
  auto &op_seq_lower_info = _lower_info_map.op_seq;
  assert(op_seq_lower_info.find(op_seq_index) != op_seq_lower_info.end());
  for (auto it = op_seq_lower_info.begin(); it != op_seq_lower_info.end(); ++it)
  {
    if (it->first == op_seq_index)
    {
      op_seq_lower_info.erase(it);
      break;
    }
  }
}

const ir::operand::LowerInfo *LoweredGraph::getLowerInfo(const ir::OperandIndex &index) const
{
  auto itr = _lower_info_map.operand.find(index);
  if (itr == _lower_info_map.operand.end())
    return nullptr;
  return itr->second.get();
}

ir::operand::LowerInfo *LoweredGraph::getLowerInfo(const ir::OperandIndex &index)
{
  auto itr = _lower_info_map.operand.find(index);
  if (itr == _lower_info_map.operand.end())
    return nullptr;
  return itr->second.get();
}

void LoweredGraph::setLowerInfo(const ir::OperandIndex &index,
                                std::unique_ptr<ir::operand::LowerInfo> &&lower_info)
{
  _lower_info_map.operand.insert(std::make_pair(index, std::move(lower_info)));
}

void LoweredGraph::removeLowerInfo(const ir::OperandIndex &index)
{
  _lower_info_map.operand.erase(index);
}

void LoweredGraph::iterateTopolOpSeqs(
  const std::function<void(const ir::OpSequenceIndex &, const ir::OpSequence &)> &fn) const
{
  // Topological Sorting for ir::OpSequences
  std::vector<ir::OpSequenceIndex> topol_sorted;
  ir::PostDfsIterator<true>{}.iterateOpSeqs(
    *this, [&](const ir::OpSequenceIndex &index, const ir::OpSequence &) {
      topol_sorted.emplace_back(index);
    });
  std::reverse(topol_sorted.begin(), topol_sorted.end());
  for (const auto op_seq_idx : topol_sorted)
  {
    const auto &op_seq = _op_seqs.at(op_seq_idx);
    fn(op_seq_idx, op_seq);
  }
}

void LoweredGraph::iterateTopolOpSeqs(
  const std::function<void(const ir::OpSequenceIndex &, ir::OpSequence &)> &fn)
{
  // Topological Sorting for ir::OpSequences
  std::vector<ir::OpSequenceIndex> topol_sorted;
  ir::PostDfsIterator<false>{}.iterateOpSeqs(
    *this,
    [&](const ir::OpSequenceIndex &index, ir::OpSequence &) { topol_sorted.emplace_back(index); });
  std::reverse(topol_sorted.begin(), topol_sorted.end());
  for (const auto op_seq_idx : topol_sorted)
  {
    auto &op_seq = _op_seqs.at(op_seq_idx);
    fn(op_seq_idx, op_seq);
  }
}

ir::OpSequenceIndex LoweredGraph::appendFreshSingleOpSequence(const ir::OperationIndex &node_index,
                                                              const ir::Operation &node)
{
  // Create a fresh op_seq with one operation, and append it to op_seqs
  // Create a fresh op_seq
  auto op_seq = std::make_unique<ir::OpSequence>(_graph.layout());

  // Add an operation
  op_seq->appendOperation(node_index);

  // Update input/output
  op_seq->setOutputs(node.getOutputs());
  op_seq->setInputs(node.getInputs());

  return _op_seqs.emplace(std::move(op_seq));
}

void LoweredGraph::makeOpSequences(
  ir::OperandIndexMap<std::unique_ptr<ir::operand::LowerInfo>> &operands_lower_info,
  const CompilerOptions &options, const BackendResolver &backend_resolver)
{
  // if SUBG_MAX_NODE == 0, no limit on nodes of a op_seq
  const int op_seq_max_node = options.op_seq_max_node;
  assert(op_seq_max_node >= 0);

  bool is_profiling = options.he_profiling_mode;
  ir::OpSequence *op_seq = nullptr;
  ir::OpSequenceIndex op_seq_index;

  // NOTE: The below method appends nodes while making one op_seq if needed. If something better
  // ways, happy to update this code.
  ir::PostDfsConstIterator{}.iterate(_graph, [&](const ir::OperationIndex &node_index,
                                                 const ir::Operation &node) {
    // LowerInfo for in/output operands
    auto backend = backend_resolver.getBackend(node_index);

    // Get frontend's layout
    auto frontend_layout = _graph.layout();

    // The layout of each backend should be set at another place
    // TODO Change setting layout of each backend at another place
    auto backend_layout = backend->config()->supportLayout(node, frontend_layout);

    for (auto operand : node.getInputs() | ir::Remove::UNDEFINED)
    {
      auto &&lower_info = operands_lower_info.at(operand);
      lower_info->addUsePermuteFactor(ir::operand::PermuteFactor{backend, backend_layout});
    }
    for (auto operand : node.getOutputs() | ir::Remove::UNDEFINED)
    {
      auto &&lower_info = operands_lower_info.at(operand);
      lower_info->addDefPermuteFactor(ir::operand::PermuteFactor{backend, backend_layout});
    }

    bool new_op_seq =
      (op_seq == nullptr || (op_seq_max_node != 0 &&
                             op_seq->operations().size() >= static_cast<size_t>(op_seq_max_node)));

    // for profiling each op_seq must contain just one node,
    // so that we can measure a node separately
    if (new_op_seq || is_profiling ||
        !mergeable(op_seq_index, node_index, backend_layout, backend_resolver))
    {
      auto new_op_seq_index = appendFreshSingleOpSequence(node_index, node);

      // ir::OpSequence LowerInfo
      setLowerInfo(new_op_seq_index,
                   std::make_unique<ir::operation::LowerInfo>(backend, backend_layout));

      op_seq_index = new_op_seq_index;
      op_seq = &(_op_seqs.at(new_op_seq_index));

      VERBOSE(Lower) << op_seq_index << " is created for " << node_index << "(" << node.name()
                     << ")" << std::endl;
    }
    else
    {
      op_seq->appendOperation(node_index);
      // Set inputs
      auto new_inputs = node.getInputs();
      // Add inputs except outputs of the previous node
      for (auto ind : op_seq->getInputs())
      {
        if (!node.getOutputs().contains(ind))
          new_inputs.append(ind);
      }
      op_seq->setInputs(new_inputs);

      VERBOSE(Lower) << op_seq_index << " merges " << node_index << "(" << node.name() << ")"
                     << std::endl;
    }
  });
}

void LoweredGraph::manipulateLowerInfo(
  ir::OperandIndexMap<std::unique_ptr<ir::operand::LowerInfo>> &operands_lower_info)
{
  const auto controlflow_backend = BackendManager::get().getControlflow();

  // TODO Rather than using NHWC Get frontend layout of this node from IR
  auto factor = ir::operand::PermuteFactor{controlflow_backend, ir::Layout::NHWC};
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
      lower_info->addDefPermuteFactor(ir::operand::PermuteFactor{
        controlflow_backend,
        ir::Layout::NHWC // TODO Get frontend layout of this node from IR
      });
    }
  }

  // 1. Add def of variable operand
  // 2. Set LowerInfo for each operand from the operand::LowerInfo holder
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
      auto factors_to_string = [](const ir::operand::PermuteFactorSet &factors) {
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
      VERBOSE(Lower) << e.second;
    }
  }
}

bool LoweredGraph::mergeable(const ir::OpSequenceIndex &op_seq_index,
                             const ir::OperationIndex &node_index, ir::Layout layout,
                             const BackendResolver &backend_resolver)
{
  // Are they mergeable?
  // 1. the same backend id and layout?
  // 2. Is op_seq or node branched?
  // 3. if 1 is true, the op_seq and a node are connected?
  const auto &op_seq = _op_seqs.at(op_seq_index);
  const auto &node = _graph.operations().at(node_index);

  // The same backend id and layout?
  {
    const auto op_seq_backend_layout = getLowerInfo(op_seq_index)->layout();
    const auto &op_seq_backend_id = getLowerInfo(op_seq_index)->backend()->config()->id();
    const auto &node_backend_id = backend_resolver.getBackend(node_index)->config()->id();
    VERBOSE(Lower) << "OpSequence" << op_seq_index << " { " << op_seq_backend_id << "("
                   << to_string(op_seq_backend_layout) << ") } "
                   << " NODE" << node_index << " (" << node.name() << ") { " << node_backend_id
                   << "(" << to_string(layout) << ") } " << std::endl;
    if (op_seq_backend_id != node_backend_id || op_seq_backend_layout != layout)
      return false;
  }

  // Branched?
  {
    std::unordered_set<ir::OperationIndex> branched_set;

    // Check for branching up
    for (const auto &input : op_seq.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      const auto &input_obj = _graph.operands().at(input);
      auto def = input_obj.getDef();
      if (def.valid())
      {
        branched_set.insert(def);
        if (branched_set.size() > 1)
        {
          return false;
        }
      }
    }
    branched_set.clear();

    // Check for branching down
    for (const auto &output : node.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      // TODO Fix this workaround for the case of model outputs that are used by another operation
      //      This is needed since the branching is decided by operation, but for model outputs,
      //      there is controlflow backen(use backend) but no actual use operation exists
      if (_graph.getOutputs().contains(output))
        return false;

      const auto &output_obj = _graph.operands().at(output);
      for (const auto &use : output_obj.getUses())
      {
        branched_set.insert(use);
        if (branched_set.size() > 1)
        {
          return false;
        }
      }
    }
  }

  // Connected?
  // an input of one node is an output of the other node? or vice-versa?
  {
    const auto &node_inputs = node.getInputs();
    const auto &node_outputs = node.getOutputs();

    // op_seq's operations are in order so that we just check the first and the last
    std::vector<ir::OperationIndex> op_seq_ops{op_seq.operations()[0]};
    if (op_seq.operations().size() > 1)
      op_seq_ops.emplace_back(op_seq.operations()[op_seq.operations().size() - 1]);

    for (const auto &n_index : op_seq_ops)
    {
      const auto &n = _graph.operations().at(n_index);

      // node's output == op_seq's input?
      for (const auto input : n.getInputs() | ir::Remove::UNDEFINED)
      {
        if (node_outputs.contains(input))
        {
          VERBOSE(Lower) << "OpSequence" << op_seq_index << " 's NODE" << n_index.value() << "("
                         << n.name() << ") is connected to NODE" << node_index.value() << "("
                         << node.name() << ")" << std::endl;
          return true;
        }
      }

      // node's input == op_seq's output?
      for (const auto output : n.getOutputs() | ir::Remove::UNDEFINED)
      {
        if (node_inputs.contains(output))
        {
          VERBOSE(Lower) << "OpSequence" << op_seq_index << " 's NODE" << n_index.value() << " ("
                         << n.name() << ") is connected to NODE" << node_index.value() << std::endl;
          return true;
        }
      }
    }

    VERBOSE(Lower) << "OpSequence" << op_seq_index << " is not connected to NODE" << node_index
                   << "(" << node.name() << ")" << std::endl;
  }

  return false;
}

} // namespace compiler
} // namespace onert
