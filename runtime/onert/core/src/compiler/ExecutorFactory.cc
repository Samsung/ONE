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

#include <deque>
#include <functional>
#include "exec/ExecutionObservers.h"
#include "exec/LinearExecutor.h"
#include "exec/DataflowExecutor.h"
#include "exec/ParallelExecutor.h"
#include "compiler/BackendManager.h"
#include "compiler/ExecutionBuilder.h"
#include "exec/ExecTime.h"
#include "compiler/Linear.h"
#include "compiler/BackendManager.h"
#include "backend/IPortableTensor.h"
#include "backend/builtin/Config.h"
#include "backend/builtin/KernelGenerator.h"
#include "backend/builtin/UserTensor.h"
#include "backend/builtin/TensorBuilder.h"
#include "util/TracingCtx.h"

#include <memory>

namespace onert
{
namespace
{

class SyncFunction final : public exec::IFunction
{
public:
  virtual ~SyncFunction() = default;
  SyncFunction(std::unique_ptr<exec::IFunction> fn, const std::shared_ptr<backend::IConfig> config)
    : _fn{std::move(fn)}, _config{config}
  {
    assert(_fn);
    assert(_config);
  }

  void run() override
  {
    _fn->run();
    _config->sync();
  }

  void prepare() override { _fn->prepare(); }

private:
  std::unique_ptr<exec::IFunction> _fn;
  std::shared_ptr<backend::IConfig> _config;
};

void initializeSubgraphIOTensors(compiler::LoweredGraph &lowered_graph,
                                 const backend::BackendContexts &backend_contexts,
                                 const ir::OperandIndexSequence &indices)
{
  // TODO Store builtin backend in BackendContext
  std::shared_ptr<backend::builtin::TensorRegistry> builtin_tensor_reg;
  for (const auto &e : backend_contexts)
  {
    auto backend = e.first;
    auto &context = e.second;
    if (backend->config()->id() == backend::builtin::Config::ID)
    {
      builtin_tensor_reg =
        std::dynamic_pointer_cast<backend::builtin::TensorRegistry>(context->tensor_registry);
    }
  }
  assert(builtin_tensor_reg);

  for (auto ind : indices)
  {
    const auto &operand = lowered_graph.graph().operands().at(ind);
    auto tensor = std::make_unique<backend::builtin::IOTensor>(
      operand.info(),
      ir::Layout::NHWC /* FIXME find operation for this operand and use frontend_layout */
    );

    // Add tensor to builtin TensorRegistry.
    builtin_tensor_reg->setNativeIOTensor(ind, std::move(tensor));
  }
}

backend::BackendContexts createBackendContexts(const ir::Graph &graph,
                                               const compiler::CompilerOptions &options)
{
  backend::BackendContexts contexts;
  bool linear_executor = (options.executor == "Linear");
  auto &backend_manager = compiler::BackendManager::get();

  // Always create Builtin backend context
  auto builtin_backend = backend_manager.getBuiltin();
  contexts.emplace(builtin_backend,
                   builtin_backend->newContext(graph, graph.getKernelBuilder(), linear_executor));

  // Create contexts for other backends
  for (auto backend_str : options.backend_list)
  {
    auto backend = backend_manager.get(backend_str);
    if (!backend)
    {
      VERBOSE_F() << "Failed to create a context - backend '" << backend_str << "' was not loaded."
                  << std::endl;
      continue;
    }

    contexts.emplace(backend,
                     backend->newContext(graph, graph.getKernelBuilder(), linear_executor));
  }
  return contexts;
}

} // namespace
} // namespace onert

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
  _map["Dataflow"] = std::bind(createDataflowExecutor, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3, false);
  _map["Parallel"] = std::bind(createDataflowExecutor, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3, true);
}

exec::IExecutor *ExecutorFactory::create(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                                         const compiler::CompilerOptions &options,
                                         const std::shared_ptr<exec::ExecutorMap> &executor_map)
{
  return _map.at(options.executor)(std::move(lowered_graph), options, executor_map);
}

void ExecutorFactory::initializeBackendContext(compiler::LoweredGraph *lowered_graph,
                                               const backend::BackendContexts &backend_contexts)
{
  struct Entry
  {
    std::vector<backend::BackendContext::OperationInfo> operation_list;
    std::vector<ir::OperandIndex> operand_list;
  };
  std::unordered_map<const backend::Backend *, Entry> backend_assets;

  lowered_graph->graph().operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::Operation &) {
      auto &op_li = lowered_graph->lower_info().operation;
      auto backend = op_li.getRawPtr(op_ind)->backend();
      backend_assets[backend].operation_list.emplace_back(op_ind, lowered_graph->graph().layout());
    });

  // Build lists for operands
  lowered_graph->graph().operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    const auto lower_info = lowered_graph->lower_info().operand.getRawPtr(ind);
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
    backend_contexts.at(backend)->initialize(arg.operation_list, arg.operand_list);
  }
}

void ExecutorFactory::prepareMigrantTensors(compiler::LoweredGraph &lowered_graph,
                                            const backend::BackendContexts &backend_contexts)
{
  TensorRegistries tensor_regs{backend_contexts, true};

  lowered_graph.graph().operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::Operation &op) {
      auto lower_info = lowered_graph.lower_info().operation.getRawPtr(op_ind);
      auto &backend_ctx = backend_contexts.at(lower_info->backend());
      for (auto ind :
           (op.getInputs() + op.getOutputs()) | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
      {
        // If an Operation's input/output tensor does not have an own tensor object,
        // it must be using migrant tensors, so find the tensor from other tensor registries and
        // register it to the current tensor registry if it is portable
        if (!backend_ctx->tensor_registry->getITensor(ind))
        {
          auto tensor = tensor_regs.getITensor(ind);
          assert(tensor); // The tensor must have been registered
          auto ptensor = dynamic_cast<backend::IPortableTensor *>(tensor);
          if (ptensor)
            backend_ctx->tensor_registry->setMigrantTensor(ind, ptensor);
        }
      }
    });
}

exec::IExecutor *
ExecutorFactory::createLinearExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                                      const compiler::CompilerOptions &options,
                                      const std::shared_ptr<exec::ExecutorMap> &executor_map)
{
  backend::BackendContexts backend_contexts =
    createBackendContexts(lowered_graph->graph(), options);
  initializeBackendContext(lowered_graph.get(), backend_contexts);

  TensorRegistries tensor_regs{backend_contexts, true};

  assert(!lowered_graph->graph().isBuildingPhase());

  initializeSubgraphIOTensors(
    *lowered_graph, backend_contexts,
    (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
      ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  // linearize
  auto order = Linear::linearize(*lowered_graph);
  Linear::dump(*lowered_graph, order);

  for (auto &pair : backend_contexts)
  {
    pair.second->genTensors(order, lowered_graph->lower_info());
  }

  prepareMigrantTensors(*lowered_graph, backend_contexts);

  // Give some runtime objects to builtin KernelGenerator
  for (auto &pair : backend_contexts)
  {
    auto builtin_context = dynamic_cast<backend::builtin::BackendContext *>(pair.second.get());
    if (builtin_context != nullptr)
    {
      auto builtin_kernel_gen = builtin_context->kernel_gen;
      builtin_kernel_gen->setTensorRegistries(tensor_regs);
      builtin_kernel_gen->setExecutorMap(executor_map);
    }
  }

  ExecutionBuilder builder;

  // Adjust the order of backends for the upcoming iteration
  std::deque<std::pair<const backend::Backend *, backend::BackendContext *>> ordered_contexts;
  for (auto &pair : backend_contexts)
  {
    // NOTE builtin backend must be processed lastly.
    // This is because of Permute layer's specialty which is the only operation that could have
    // different ITensor objects for the input and the output. And it requires all other backends'
    // tensors are ready to use.
    if (pair.first->config()->id() == "builtin")
      ordered_contexts.emplace_back(pair.first, pair.second.get());
    else
      ordered_contexts.emplace_front(pair.first, pair.second.get());
  }

  // Generate kernels
  for (auto &pair : ordered_contexts)
  {
    auto codes = pair.second->genKernels(order);
    for (auto &pair : codes)
    {
      auto &op_ind = pair.first;
      auto &fn_seq = pair.second;
      auto &op = lowered_graph->graph().operations().at(op_ind);
      auto lower_info = lowered_graph->lower_info().operation.getRawPtr(op_ind);
      if (options.he_profiling_mode)
        fn_seq->wrap<SyncFunction>(lower_info->backend()->config());
      builder.append(op_ind, {op_ind, &op, lower_info, std::move(fn_seq)});
    }
  }

  auto code_map = builder.releaseCodeMap();

  auto exec = new exec::LinearExecutor{
    std::move(lowered_graph), std::move(backend_contexts), tensor_regs, std::move(code_map), order,
    options.tracing_ctx};

  if (!options.trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp = std::make_unique<exec::TracingObserver>(
      options.trace_filepath, exec->graph(), options.tracing_ctx);
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

exec::IExecutor *ExecutorFactory::createDataflowExecutor(
  std::unique_ptr<compiler::LoweredGraph> lowered_graph, const compiler::CompilerOptions &options,
  const std::shared_ptr<exec::ExecutorMap> &executor_map, bool parallel)
{
  backend::BackendContexts backend_contexts =
    createBackendContexts(lowered_graph->graph(), options);

  initializeBackendContext(lowered_graph.get(), backend_contexts);

  TensorRegistries tensor_regs{backend_contexts, true};

  assert(!lowered_graph->graph().isBuildingPhase());

  initializeSubgraphIOTensors(
    *lowered_graph, backend_contexts,
    (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
      ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  // linearize
  // This order is just for giving topological order info to the backens
  // TODO When we pass a partial graph to a backend, we can remove this
  auto order = Linear::linearize(*lowered_graph);
  for (auto &pair : backend_contexts)
  {
    pair.second->genTensors(order, lowered_graph->lower_info());
  }

  prepareMigrantTensors(*lowered_graph, backend_contexts);

  // Give some runtime objects to builtin KernelGenerator
  for (auto &pair : backend_contexts)
  {
    auto builtin_context = dynamic_cast<backend::builtin::BackendContext *>(pair.second.get());
    if (builtin_context != nullptr)
    {
      auto builtin_kernel_gen = builtin_context->kernel_gen;
      builtin_kernel_gen->setTensorRegistries(tensor_regs);
      builtin_kernel_gen->setExecutorMap(executor_map);
    }
  }

  ExecutionBuilder builder;

  // Adjust the order of backends for the upcoming iteration
  std::deque<std::pair<const backend::Backend *, backend::BackendContext *>> ordered_contexts;
  for (auto &pair : backend_contexts)
  {
    // NOTE builtin backend must be processed lastly.
    // This is because of Permute layer's specialty which is the only operation that could have
    // different ITensor objects for the input and the output. And it requires all other backends'
    // tensors are ready to use.
    if (pair.first->config()->id() == "builtin")
      ordered_contexts.emplace_back(pair.first, pair.second.get());
    else
      ordered_contexts.emplace_front(pair.first, pair.second.get());
  }

  // Generate kernels
  for (auto &pair : ordered_contexts)
  {
    auto codes = pair.second->genKernels(order);
    for (auto &pair : codes)
    {
      auto &op_ind = pair.first;
      auto &fn_seq = pair.second;
      auto &op = lowered_graph->graph().operations().at(op_ind);
      auto lower_info = lowered_graph->lower_info().operation.getRawPtr(op_ind);
      if (options.he_profiling_mode)
        fn_seq->wrap<SyncFunction>(lower_info->backend()->config());
      builder.append(op_ind, {op_ind, &op, lower_info, std::move(fn_seq)});
    }
  }

  auto code_map = builder.releaseCodeMap();

  exec::ExecutorBase *exec = nullptr;
  if (parallel)
  {
    exec = new exec::ParallelExecutor{std::move(lowered_graph), std::move(backend_contexts),
                                      tensor_regs, std::move(code_map), options.tracing_ctx};
  }
  else
  {
    auto dataflow_exec =
      new exec::DataflowExecutor{std::move(lowered_graph), std::move(backend_contexts), tensor_regs,
                                 std::move(code_map), options.tracing_ctx};
    if (options.he_profiling_mode)
    {
      std::vector<const backend::Backend *> backends;
      for (const auto &pair : backend_contexts)
      {
        backends.push_back(pair.first);
      }
      auto et = std::make_shared<exec::ExecTime>(backends);
      std::unique_ptr<exec::IExecutionObserver> obs =
        std::make_unique<exec::ProfileObserver>(et, dataflow_exec->graph());
      dataflow_exec->addObserver(std::move(obs));
    }
    exec = dataflow_exec;
  }

  if (!options.trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp = std::make_unique<exec::TracingObserver>(
      options.trace_filepath, exec->graph(), options.tracing_ctx);
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

} // namespace compiler
} // namespace onert
