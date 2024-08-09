/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "compiler/train/LoweredTrainableGraph.h"

#include "../ManualScheduler.h"
#include "../pass/ConstantInsertionPass.h"
#include "../pass/ConstantLoweringPass.h"
#include "../pass/PassRunner.h"
#include "../pass/PermutationEliminationPass.h"
#include "../pass/PermutationInsertionPass.h"
#include "../../backend/builtin/Config.h"
#include "../../dumper/text/GraphDumper.h"
#include "../../ir/verifier/Verifier.h"
#include "pass/TrainableConstantInsertionPass.h"
#include "TrainableOperationConverter.h"

#include "backend/Backend.h"
#include "backend/train/ITrainableBackend.h"
#include "compiler/BackendResolver.h"
#include "util/logging.h"

#include <cassert>
#include <sstream>

namespace onert
{
namespace compiler
{
namespace train
{

LoweredTrainableGraph::LoweredTrainableGraph(ir::train::TrainableGraph &graph,
                                             const CompilerOptions &options)
  : _trainable_graph{graph}
{
  lowerGraph(options);
}

void LoweredTrainableGraph::lowerGraph(const CompilerOptions &options)
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
      VERBOSE(LoweredTrainableGraph) << "Cannot load backend - " << backend_str << std::endl;
      continue;
    }
  }
  if (backend_manager.num_backends() == 0)
    throw std::runtime_error{"No available backends loaded."};

  // TODO Move "schedule" phase out of here
  // TODO Scheduling
  std::unique_ptr<BackendResolver> backend_resolver;
  auto all_backends = backend_manager.getAll();

  auto scheduler = ManualScheduler(all_backends, options);
  backend_resolver = scheduler.schedule(_trainable_graph.graph());

  // Check if backends are trainable
  _trainable_graph.operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::IOperation &) {
      const auto backend = backend_resolver->getBackend(op_ind);

      // TODO Remove dynamic_cast
      if (dynamic_cast<const backend::train::ITrainableBackend *>(backend) == nullptr)
      {
        throw std::runtime_error(backend->config()->id() + "backend does not support training");
      }
    });

  makeLowerInfo(*backend_resolver);
  VERBOSE(LoweredTrainableGraph) << "dump before mandatory passes" << std::endl;
  dumper::text::dumpLoweredGraph(*this);

  // Mandatory passes - kind of legalization(?)
  compiler::pass::PassRunner{}
    .append(std::make_unique<compiler::pass::ConstantInsertionPass>(*this))
    .append(std::make_unique<compiler::train::pass::TrainableConstantInsertionPass>(*this))
    .append(std::make_unique<compiler::pass::ConstantLoweringPass>(*this))
    .append(std::make_unique<compiler::pass::PermutationInsertionPass>(*this))
    .run();

  // TODO Move converting Permute op into PermutationInsertionPass
  auto op_converter = TrainableOperationConverter{_trainable_graph, nullptr};
  _trainable_graph.operations().iterate(
    [&](const onert::ir::OperationIndex &index, const onert::ir::IOperation &op) {
      if (op.opcode() == ir::OpCode::Permute)
      {
        auto trainable_op = op_converter(op);
        trainable_op->enableBackward();
        auto gen_index = _trainable_graph.replaceOperation(index, std::move(trainable_op));
        UNUSED_RELEASE(gen_index);
        assert(gen_index == index);
      }
    });

  dumpLowerInfo();

  // Optimization passes (optional)
  compiler::pass::PassRunner{}
    .append(std::make_unique<compiler::pass::PermutationEliminationPass>(*this))
    .run();

  // TODO Update LowerInfo for training

  VERBOSE(LoweredTrainableGraph) << "Dump after all the passes" << std::endl;
  for (auto &&operand : _trainable_graph.getInputs())
    VERBOSE(LoweredTrainableGraph) << "Graph Input : " << operand << std::endl;
  for (auto &&operand : _trainable_graph.getOutputs())
    VERBOSE(LoweredTrainableGraph) << "Graph Output : " << operand << std::endl;
  dumper::text::dumpLoweredGraph(*this);

  // Graph verifications
  {
    assert(ir::verifier::InputOutputChecker().verify(_trainable_graph.graph()));
    assert(ir::verifier::DAGChecker().verify(_trainable_graph.graph()));
    assert(ir::verifier::EdgeChecker().verify(_trainable_graph.graph()));
  }
}

void LoweredTrainableGraph::makeLowerInfo(const compiler::BackendResolver &backend_resolver)
{
  _trainable_graph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &) {
    lower_info().operand.set(index, std::make_unique<OperandLowerInfo>());
  });

  // Set operand lower info using assigned backends to operations
  _trainable_graph.operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::IOperation &op) {
      auto backend = backend_resolver.getBackend(op_ind);
      if (!backend)
      {
        throw std::runtime_error{"Fail to find backend for " + op.name() + " operation"};
      }

      for (auto &&ind : op.getInputs() | ir::Remove::UNDEFINED)
      {
        auto &operand_li = lower_info().operand.at(ind);
        operand_li.addUseBackend(backend);
      }
      for (auto &&ind : op.getOutputs() | ir::Remove::UNDEFINED)
      {
        auto &operand_li = lower_info().operand.at(ind);
        operand_li.addDefBackend(backend);
      }
      lower_info().operation.emplace(op_ind, backend);
    });

  // Handle graph inputs and outputs
  const auto builtin_backend = BackendManager::get().getBuiltin();
  for (auto &&index : _trainable_graph.getInputs() | ir::Remove::UNDEFINED)
  {
    auto &operand_li = lower_info().operand.at(index);
    assert(operand_li.def_backends().empty());
    operand_li.addDefBackend(builtin_backend);
  }
  for (auto &&index : _trainable_graph.getOutputs() | ir::Remove::UNDEFINED)
  {
    auto &operand_li = lower_info().operand.at(index);
    operand_li.addUseBackend(builtin_backend);
  }

  // Handle variable tensors
  _trainable_graph.operands().iterate([&](const ir::OperandIndex &index, ir::Operand &operand) {
    // Some inputs of an operation could be non-constant, but not existed in graph inputs/outputs
    // and not undefined operand - these are variable tensors. For example,
    // UnidirectionalSequenceLSTM has such inputs.
    if (operand.info().isVariable())
    {
      // The variable operand with buffer is not supported yet
      assert(operand.data() == nullptr);
      assert(operand.getUses().size() == 1 && !operand.getDef().valid());
      auto operand_li = lower_info().operand.at(index);
      assert(operand_li.def_backends().empty());
      operand_li.addDefBackend(operand_li.use_backends().getOnlyElement());
    }
  });
}

void LoweredTrainableGraph::dumpLowerInfo()
{
  if (::onert::util::logging::ctx.enabled() == false)
    return;

  std::map<uint32_t, std::string> dumps;

  _trainable_graph.operands().iterate([&](const ir::OperandIndex &index, ir::Operand &object) {
    const auto operand_lower_info = lower_info().operand.getRawPtr(index);
    assert(operand_lower_info);
    if (!operand_lower_info->def_backends().empty() || !operand_lower_info->use_backends().empty())
    {
      auto shape_to_string = [](const ir::Shape &shape) {
        std::stringstream sstream;
        sstream << "{ ";
        for (auto i = 0; i < shape.rank(); ++i)
          sstream << (shape.dim(i)) << " ";
        sstream << "}";
        return sstream.str();
      };

      auto backends_to_string = [](const BackendSet &backends) {
        std::string str;
        for (auto &&backend : backends)
        {
          str += backend->config()->id();
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
      std::string def_backends = backends_to_string(operand_lower_info->def_backends());
      std::string use_backends = backends_to_string(operand_lower_info->use_backends());
      std::stringstream sstream;
      sstream << "Operand " << index << " Info" << std::endl;
      sstream << "  - Shape     : " << shape_str << std::endl;
      sstream << "  - Def/Uses  : Def " << def_op << " Uses " << use_ops << std::endl;
      sstream << "  - Data      : " << data_to_str(object.data()) << std::endl;
      sstream << "  - LowerInfo : Def " << def_backends << " Uses " << use_backends << std::endl;
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

} // namespace train
} // namespace compiler
} // namespace onert
