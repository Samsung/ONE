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

#include "HEScheduler.h"
#include "ManualScheduler.h"
#include "pass/ConstantInsertionPass.h"
#include "pass/ConstantLoweringPass.h"
#include "pass/PassRunner.h"
#include "pass/PermutationEliminationPass.h"
#include "pass/PermutationInsertionPass.h"
#include "pass/PermutationOperationPass.h"
#include "../dumper/text/GraphDumper.h"
#include "../ir/verifier/Verifier.h"

#include "backend/Backend.h"
#include "compiler/BackendResolver.h"
#include "util/logging.h"

#include <cassert>
#include <sstream>

namespace onert
{
namespace compiler
{

LoweredGraph::LoweredGraph(const ir::Graph &graph, const CompilerOptions &options) : _graph{graph}
{
  lowerGraph(options);
}

void LoweredGraph::lowerGraph(const CompilerOptions &options)
{
  // Build backend contexts
  auto &backend_manager = BackendManager::get();
  // Create contexts for other backends
  for (auto &&backend_str : options.backend_list)
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

  makeLowerInfo(*backend_resolver);
  VERBOSE(LoweredGraph) << "dump before mandatory passes" << std::endl;
  dumper::text::dumpLoweredGraph(*this);

  // Mandatory passes - kind of legalization(?)
  pass::PassRunner{}
    .append(std::make_unique<pass::ConstantInsertionPass>(*this))
    .append(std::make_unique<pass::ConstantLoweringPass>(*this))
    .append(std::make_unique<pass::PermutationOperationPass>(*this))
    .append(std::make_unique<pass::PermutationInsertionPass>(*this))
    .run();

  dumpLowerInfo();

  // Optimization passes (optional)
  pass::PassRunner{}.append(std::make_unique<pass::PermutationEliminationPass>(*this)).run();

  VERBOSE(LoweredGraph) << "Dump after all the passes" << std::endl;
  for (auto &&operand : _graph.getInputs())
    VERBOSE(LoweredGraph) << "Graph Input : " << operand << std::endl;
  for (auto &&operand : _graph.getOutputs())
    VERBOSE(LoweredGraph) << "Graph Output : " << operand << std::endl;
  dumper::text::dumpLoweredGraph(*this);

  // Graph verifications
  {
    assert(ir::verifier::InputOutputChecker().verify(_graph));
    assert(ir::verifier::DAGChecker().verify(_graph));
    assert(ir::verifier::EdgeChecker().verify(_graph));
  }
}

void LoweredGraph::makeLowerInfo(const compiler::BackendResolver &backend_resolver)
{
  _graph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &) {
    lower_info().operand.set(index, std::make_unique<OperandLowerInfo>());
  });

  // Set operand lower info using assigned backends to operations
  _graph.operations().iterate([&](const ir::OperationIndex &op_ind, const ir::IOperation &) {
    const ir::IOperation &op = _graph.operations().at(op_ind);
    auto backend = backend_resolver.getBackend(op_ind);
    if (!backend)
    {
      throw std::runtime_error{"Fail to find backend for " + op.name() + " operation"};
    }

    auto frontend_layout = _graph.layout();

    // The layout of each backend should be set at another place
    // TODO Change setting layout of each backend at another place
    auto backend_layout = backend->config()->supportLayout(op, frontend_layout);

    for (auto &&ind : op.getInputs() | ir::Remove::UNDEFINED)
    {
      auto &operand_li = lower_info().operand.at(ind);
      operand_li.addUsePermuteFactor(PermuteFactor{backend, backend_layout});
    }
    for (auto &&ind : op.getOutputs() | ir::Remove::UNDEFINED)
    {
      auto &operand_li = lower_info().operand.at(ind);
      operand_li.addDefPermuteFactor(PermuteFactor{backend, backend_layout});
    }
    lower_info().operation.set(
      op_ind, std::make_unique<compiler::OperationLowerInfo>(backend, backend_layout));
  });

  // Handle graph inputs and outputs
  const auto builtin_backend = BackendManager::get().getBuiltin();
  auto factor = PermuteFactor{builtin_backend, _graph.layout()};
  for (auto &&index : _graph.getInputs() | ir::Remove::UNDEFINED)
  {
    auto &operand_li = lower_info().operand.at(index);
    assert(operand_li.def_factors().empty());
    operand_li.addDefPermuteFactor(factor);
  }
  for (auto &&index : _graph.getOutputs() | ir::Remove::UNDEFINED)
  {
    auto &operand_li = lower_info().operand.at(index);
    operand_li.addUsePermuteFactor(factor);
  }

  // Handle variable tensors
  _graph.operands().iterate([&](const ir::OperandIndex &index, ir::Operand &operand) {
    // Some inputs of an operation could be non-constant, but not existed in graph inputs/outputs
    // and not undefined operand - these are variable tensors. For example,
    // UnidirectionalSequenceLSTM has such inputs.
    if (operand.info().isVariable())
    {
      // The variable operand with buffer is not supported yet
      assert(operand.data() == nullptr);
      assert(operand.getUses().size() == 1 && !operand.getDef().valid());
      auto operand_li = lower_info().operand.at(index);
      assert(operand_li.def_factors().empty());
      operand_li.addDefPermuteFactor(operand_li.use_factors().getOnlyElement());
    }
  });
}

void LoweredGraph::dumpLowerInfo()
{
  if (::onert::util::logging::ctx.enabled() == false)
    return;

  std::map<uint32_t, std::string> dumps;

  _graph.operands().iterate([&](const ir::OperandIndex &index, ir::Operand &object) {
    const auto operand_lower_info = lower_info().operand.getRawPtr(index);
    assert(operand_lower_info);
    if (!operand_lower_info->def_factors().empty() || !operand_lower_info->use_factors().empty())
    {
      auto shape_to_string = [](const ir::Shape &shape) {
        std::stringstream sstream;
        sstream << "{ ";
        for (auto i = 0; i < shape.rank(); ++i)
          sstream << (shape.dim(i)) << " ";
        sstream << "}";
        return sstream.str();
      };

      auto factors_to_string = [](const PermuteFactorSet &factors) {
        std::string str;
        for (auto &&factor : factors)
        {
          str += factor.backend()->config()->id();
          str += "(" + to_string(factor.layout()) + ")";
          str += " ";
        }
        return "{ " + str + "}";
      };

      auto operation_index_set_to_string = [](const ir::OperationIndexSet &operations) {
        std::stringstream sstream;
        sstream << "{ ";
        for (auto &&op : operations)
          sstream << op << " ";
        sstream << "}";
        return sstream.str();
      };

      auto data_to_str = [](const ir::Data *data) {
        return (data ? (std::to_string(data->size()) + " bytes") : "N/A");
      };

      std::string shape_str = shape_to_string(object.shape());
      std::string def_op = operation_index_set_to_string({object.getDef()});
      std::string use_ops = operation_index_set_to_string(object.getUses());
      std::string def_factors = factors_to_string(operand_lower_info->def_factors());
      std::string use_factors = factors_to_string(operand_lower_info->use_factors());
      std::stringstream sstream;
      sstream << "Operand " << index << " Info" << std::endl;
      sstream << "  - Shape     : " << shape_str << std::endl;
      sstream << "  - Def/Uses  : Def " << def_op << " Uses " << use_ops << std::endl;
      sstream << "  - Data      : " << data_to_str(object.data()) << std::endl;
      sstream << "  - LowerInfo : Def " << def_factors << " Uses " << use_factors << std::endl;
      dumps.emplace(index.value(), sstream.str());
    }
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
