/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExecutorFactory.h"

#include <functional>
#include "exec/ExecutionObservers.h"
#include "exec/LinearExecutor.h"
#include "exec/DataflowExecutor.h"
#include "exec/ParallelExecutor.h"
#include "compiler/BackendManager.h"
#include "compiler/ExecutionBuilder.h"
#include "exec/ExecTime.h"
#include "compiler/Linear.h"
#include "backend/IConstantInitializer.h"
#include "backend/IKernelGenerator.h"
#include "backend/IShapeFixer.h"
#include "backend/IOptimizer.h"
#include "backend/ITensorRegister.h"
#include <memory>
#include "compiler/CachedDataDeleter.h"
#include "util/ShapeInference.h"

namespace onert
{
namespace compiler
{

ExecutorFactory &ExecutorFactory::get()
{
  static ExecutorFactory singleton;
  return singleton;
}

ExecutorFactory::ExecutorFactory()
{
  _map["Linear"] = createLinearExecutor;
  _map["Dataflow"] =
      std::bind(createDataflowExecutor, std::placeholders::_1, std::placeholders::_2, false);
  _map["Parallel"] =
      std::bind(createDataflowExecutor, std::placeholders::_1, std::placeholders::_2, true);
}

exec::IExecutor *ExecutorFactory::create(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                                         const compiler::CompilerOptions &options)
{
  return _map.at(options.executor)(std::move(lowered_graph), options);
}

void ExecutorFactory::initializeBackendContext(ir::LoweredGraph *lowered_graph)
{
  struct Entry
  {
    std::vector<backend::BackendContext::OperationInfo> operation_list;
    std::vector<ir::OperandIndex> operand_list;
  };
  std::unordered_map<const backend::Backend *, Entry> backend_assets;

  // Build lists for operations
  lowered_graph->op_seqs().iterate(
      [&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &op_seq) {
        auto &op_seq_li = lowered_graph->getLowerInfo()->operation;
        auto backend = op_seq_li.at(op_seq_index)->backend();
        for (auto &element : op_seq.operations())
        {
          backend_assets[backend].operation_list.emplace_back(element.index, op_seq.getLayout());
        }
      });

  // Build lists for operands
  lowered_graph->graph().operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    const auto lower_info = lowered_graph->getLowerInfo(ind);
    for (auto factor : lower_info->def_factors())
    {
      auto backend = factor.backend();
      backend_assets[backend].operand_list.emplace_back(ind);
    }
  });

  for (auto &pair : backend_assets)
  {
    auto backend = pair.first;
    auto &arg = pair.second;
    lowered_graph->backend_contexts().at(backend)->initialize(arg.operation_list, arg.operand_list);
  }
}

void ExecutorFactory::runTensorRegistration(ir::LoweredGraph *lowered_graph,
                                            const std::vector<ir::OpSequenceIndex> &order)
{
  for (const auto index : order)
  {
    const auto &op_seq = lowered_graph->op_seqs().at(index);
    const auto backend = lowered_graph->getLowerInfo(index)->backend();
    const auto tensor_register = lowered_graph->backend_contexts().at(backend)->tensor_register;
    auto tensor_builder = lowered_graph->backend_contexts().at(backend)->tensor_builder;
    if (tensor_register)
    {
      // Custom registration
      tensor_register->registerTensors(op_seq, lowered_graph->getLowerInfo());
    }
    else
    {
      // Default registration
      for (const auto elem : op_seq)
      {
        const auto &op = *elem.node;
        for (const auto &index : op.getInputs() + op.getOutputs())
        {
          if (!tensor_builder->isRegistered(index))
          {
            const auto &operand_lower_info =
                lowered_graph->getLowerInfo(index)->def_factors().getOnlyElement();
            const auto &obj = lowered_graph->graph().operands().at(index);
            const auto frontend_layout = op_seq.getLayout();
            const auto backend_layout = operand_lower_info.layout();
            ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                         obj.typeInfo()};
            tensor_builder->registerTensorInfo(index, backend_info, backend_layout,
                                               obj.isConstant());
          }
        }
      }
    }
  }
}

exec::IExecutor *
ExecutorFactory::createLinearExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                                      const compiler::CompilerOptions &options)
{
  const auto &backend_contexts = lowered_graph->backend_contexts();

  initializeBackendContext(lowered_graph.get());

  // linearize
  assert(!lowered_graph->graph().isBuildingPhase());

  // Shape inference.
  {
    shape_inference::StaticInferer inferer(lowered_graph->graph().operands());
    lowered_graph->op_seqs().iterate(
        [&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) { inferer.infer(op_seq); });
  }

  for (auto &pair : backend_contexts)
  {
    pair.second->fixShapes();
  }

  /*************************************************
   * Backend dependent analysis & optimization phase
   *************************************************/

  for (auto &pair : backend_contexts)
  {
    auto &optimizer = pair.second->optimizer;
    if (optimizer)
      optimizer->optimize();
  }

  /**********************************************************
   * Backend dependent analysis & optimization phase finished
   **********************************************************/

  /***********************
   * Code generation phase
   ***********************/

  auto order = Linear::linearize(*lowered_graph);
  runTensorRegistration(lowered_graph.get(), order);
  Linear::dump(*lowered_graph, order);
  Linear::planTensors(*lowered_graph, order);

  backend::TensorBuilderSet tensor_builders;
  for (const auto &e : lowered_graph->backend_contexts())
  {
    tensor_builders.insert(e.second->tensor_builder);
  }

  for (auto &tensor_builder : tensor_builders)
  {
    tensor_builder->prepare();
  }

  ExecutionBuilder builder;

  // Generate kernels
  lowered_graph->op_seqs().iterate(
      [&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &op_seq) {
        auto lower_info = lowered_graph->getLowerInfo(op_seq_index);
        auto kernel_gen = lowered_graph->backend_contexts().at(lower_info->backend())->kernel_gen;
        auto fn_seq = kernel_gen->generate(op_seq);
        builder.append(op_seq_index, {&op_seq, lower_info, std::move(fn_seq)});
      });

  for (auto &tensor_builder : tensor_builders)
  {
    tensor_builder->allocate();
  }

  for (auto &pair : backend_contexts)
  {
    pair.second->initConsts();
  }

  // Note. The best solution is not to use CachedDataDeleter but decreasing reference counts of data
  // naturally
  if (options.delete_cached_data)
  {
    CachedDataDeleter cached_data_deleter(lowered_graph->graph().operands());
    lowered_graph->op_seqs().iterate(
        [&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
          cached_data_deleter.run(op_seq);
        });
  }

  auto code_map = builder.releaseCodeMap();

  for (auto &it : code_map)
  {
    auto op_seq_index = it.first;
    auto &fn_seq = it.second.fn_seq;

    fn_seq->iterate([&](exec::IFunction &ifunc) {
      ifunc.prepare();
      auto backend = lowered_graph->getLowerInfo(op_seq_index)->backend();
      auto tensor_builder = lowered_graph->backend_contexts().at(backend)->tensor_builder;
      tensor_builder->postFunctionPrepare();
    });
  }

  auto exec = new exec::LinearExecutor{std::move(lowered_graph), tensor_builders,
                                       std::move(code_map), order};

  if (!options.trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp =
        std::make_unique<exec::ChromeTracingObserver>(options.trace_filepath);
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

exec::IExecutor *
ExecutorFactory::createDataflowExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                                        const compiler::CompilerOptions &options, bool parallel)
{
  const auto &backend_contexts = lowered_graph->backend_contexts();

  initializeBackendContext(lowered_graph.get());

  for (auto &pair : backend_contexts)
  {
    pair.second->fixShapes();
  }

  auto order = Linear::linearize(*lowered_graph);
  runTensorRegistration(lowered_graph.get(), order);

  backend::TensorBuilderSet tensor_builders;
  for (const auto &e : lowered_graph->backend_contexts())
  {
    tensor_builders.insert(e.second->tensor_builder);
  }

  // To make tensors never be deallocated, this is a workaround to use static memory planner
  for (auto &tensor_builder : tensor_builders)
  {
    lowered_graph->graph().operands().iterate(
        [&](const ir::OperandIndex &ind, const ir::Operand &) {
          if (tensor_builder->isRegistered(ind))
          {
            tensor_builder->notifyFirstUse(ind);
          }
        });
  }

  for (auto &tensor_builder : tensor_builders)
  {
    tensor_builder->prepare();
  }

  ExecutionBuilder builder;

  // Generate kernels
  lowered_graph->op_seqs().iterate(
      [&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &op_seq) {
        auto lower_info = lowered_graph->getLowerInfo(op_seq_index);
        auto kernel_gen = lowered_graph->backend_contexts().at(lower_info->backend())->kernel_gen;
        auto fn_seq = kernel_gen->generate(op_seq);
        builder.append(op_seq_index, {&op_seq, lower_info, std::move(fn_seq)});
      });

  for (const auto &tensor_builder : tensor_builders)
  {
    tensor_builder->allocate();
  }

  for (auto &pair : backend_contexts)
  {
    pair.second->initConsts();
  }

  if (options.delete_cached_data)
  {
    CachedDataDeleter cached_data_deleter(lowered_graph->graph().operands());
    lowered_graph->op_seqs().iterate(
        [&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
          cached_data_deleter.run(op_seq);
        });
  }

  auto code_map = builder.releaseCodeMap();

  for (auto &it : code_map)
  {
    auto op_seq_index = it.first;
    auto &fn_seq = it.second.fn_seq;

    fn_seq->iterate([&](exec::IFunction &ifunc) {
      ifunc.prepare();
      auto backend = lowered_graph->getLowerInfo(op_seq_index)->backend();
      auto tensor_builder = lowered_graph->backend_contexts().at(backend)->tensor_builder;
      tensor_builder->postFunctionPrepare();
    });
  }

  exec::ExecutorBase *exec = nullptr;
  if (parallel)
  {
    exec =
        new exec::ParallelExecutor{std::move(lowered_graph), tensor_builders, std::move(code_map)};
  }
  else
  {
    auto dataflow_exec =
        new exec::DataflowExecutor{std::move(lowered_graph), tensor_builders, std::move(code_map)};
    if (options.he_profiling_mode)
    {
      std::vector<const backend::Backend *> backends;
      for (const auto &pair : backend_contexts)
      {
        backends.push_back(pair.first);
      }
      auto et = std::make_shared<exec::ExecTime>(backends);
      std::unique_ptr<exec::IExecutionObserver> obs = std::make_unique<exec::ProfileObserver>(et);
      dataflow_exec->addObserver(std::move(obs));
      dataflow_exec->setProfilingMode(true);
    }
    exec = dataflow_exec;
  }

  if (!options.trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp =
        std::make_unique<exec::ChromeTracingObserver>(options.trace_filepath);
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

} // namespace compiler
} // namespace onert
