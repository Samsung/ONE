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
#include "compiler/TensorBuilders.h"
#include "backend/IConstantInitializer.h"
#include "backend/IKernelGenerator.h"
#include "backend/IOptimizer.h"
#include "backend/IPortableTensor.h"
#include "backend/controlflow/Config.h"
#include "backend/controlflow/KernelGenerator.h"
#include "backend/controlflow/UserTensor.h"
#include "backend/controlflow/TensorBuilder.h"
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
                                 const ir::OperandIndexSequence &indices)
{
  // TODO Store controlflow backend in BackendContext
  std::shared_ptr<backend::controlflow::TensorBuilder> cf_tensor_builder;
  std::shared_ptr<backend::controlflow::TensorRegistry> cf_tensor_reg;
  for (const auto &e : lowered_graph.backend_contexts())
  {
    auto backend = e.first;
    auto &context = e.second;
    if (backend->config()->id() == backend::controlflow::Config::ID)
    {
      cf_tensor_builder =
          std::dynamic_pointer_cast<backend::controlflow::TensorBuilder>(context->tensor_builder);
      cf_tensor_reg =
          std::dynamic_pointer_cast<backend::controlflow::TensorRegistry>(context->tensor_registry);
    }
  }
  assert(cf_tensor_builder);
  assert(cf_tensor_reg);

  for (auto ind : indices)
  {
    const auto &operand = lowered_graph.graph().operands().at(ind);
    auto tensor = std::make_unique<backend::controlflow::IOTensor>(
        operand.info(),
        ir::Layout::NHWC /* FIXME find op_seq for this operand and use frontend_layout */
        );

    // Add tensor to controlflow TensorRegistry.
    cf_tensor_reg->setNativeIOTensor(ind, std::move(tensor));
  }
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

void ExecutorFactory::initializeBackendContext(compiler::LoweredGraph *lowered_graph)
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
        auto &op_seq_li = lowered_graph->getLowerInfo()->op_seq;
        auto backend = op_seq_li.at(op_seq_index)->backend();
        for (auto &operation_idx : op_seq.operations())
        {
          backend_assets[backend].operation_list.emplace_back(operation_idx, op_seq.getLayout());
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

void ExecutorFactory::runTensorRegistration(compiler::LoweredGraph *lowered_graph,
                                            const std::vector<ir::OpSequenceIndex> &order)
{
  for (const auto index : order)
  {
    const auto &op_seq = lowered_graph->op_seqs().at(index);
    const auto backend = lowered_graph->getLowerInfo(index)->backend();
    auto tensor_builder = lowered_graph->backend_contexts().at(backend)->tensor_builder;
    auto model_io = lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs();

    // TODO nit: Remove indentation
    {
      // Default registration
      for (const auto op_idx : op_seq)
      {
        const auto &op = lowered_graph->graph().operations().at(op_idx);
        for (const auto &index :
             (op.getInputs() | ir::Remove::UNDEFINED) + (op.getOutputs() | ir::Remove::UNDEFINED))
        {
          if (!tensor_builder->isRegistered(index) && !model_io.contains(index))
          {
            const auto &operand_lower_info =
                lowered_graph->getLowerInfo(index)->def_factors().getOnlyElement();

            // E.g., permute (CPU) -> tensor A -> MaxPool2D(acl_cl)
            // op.getOutputs() of permute (CPU) returns tensor A
            // but tensor A belongs to the backend of acl_cl.
            // So, we have to make this tensor NOT registered for CPU.
            if (operand_lower_info.backend() != backend)
              continue;

            const auto &obj = lowered_graph->graph().operands().at(index);
            const auto frontend_layout = op_seq.getLayout();
            const auto backend_layout = operand_lower_info.layout();
            ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                         obj.typeInfo(), obj.info().memAllocType(),
                                         obj.isConstant()};
            tensor_builder->registerTensorInfo(index, backend_info, backend_layout);
          }
        }
      }
    }
  }
}

void ExecutorFactory::prepareMigrantTensors(compiler::LoweredGraph &lowered_graph)
{
  TensorRegistries tensor_regs{lowered_graph.backend_contexts(), true};

  lowered_graph.op_seqs().iterate(
      [&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &op_seq) {
        auto lower_info = lowered_graph.getLowerInfo(op_seq_index);
        auto &backend_ctx = lowered_graph.backend_contexts().at(lower_info->backend());
        for (auto ind : (op_seq.getInputs() + op_seq.getOutputs()) | ir::Remove::DUPLICATED |
                            ir::Remove::UNDEFINED)
        {
          // If an OpSequence input/output tensor does not have a own tensor object,
          // it must be using migrant tensors, so find the tensor from other tensor builders and
          // set the tensor to this tensor builder if portable
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
  const auto &backend_contexts = lowered_graph->backend_contexts();

  initializeBackendContext(lowered_graph.get());

  TensorBuilders tensor_builders{lowered_graph->backend_contexts(), true};
  TensorRegistries tensor_regs{lowered_graph->backend_contexts(), true};

  assert(!lowered_graph->graph().isBuildingPhase());

  initializeSubgraphIOTensors(
      *lowered_graph, (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
                          ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  // linearize
  auto order = Linear::linearize(*lowered_graph);
  Linear::dump(*lowered_graph, order);

  for (auto &pair : backend_contexts)
  {
    pair.second->tensorGen(order, lowered_graph->op_seqs(), *lowered_graph->getLowerInfo());
  }

  prepareMigrantTensors(*lowered_graph);

  // Give some runtime objects to controlflow KernelGenerator
  for (auto &pair : backend_contexts)
  {
    auto kernel_gen = pair.second->kernel_gen;
    auto cf_kernel_gen = dynamic_cast<backend::controlflow::KernelGenerator *>(kernel_gen.get());
    if (cf_kernel_gen != nullptr)
    {
      cf_kernel_gen->setTensorRegistries(tensor_regs);
      cf_kernel_gen->setExecutorMap(executor_map);
      break;
    }
  }

  prepareMigrantTensors(*lowered_graph);

  ExecutionBuilder builder;

  lowered_graph->iterateTopolOpSeqs(
      [&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &op_seq) {
        auto lower_info = lowered_graph->getLowerInfo(op_seq_index);
        auto kernel_gen = lowered_graph->backend_contexts().at(lower_info->backend())->kernel_gen;
        auto fn_seq = kernel_gen->generate(op_seq);
        if (options.he_profiling_mode)
        {
          fn_seq->wrap<SyncFunction>(lower_info->backend()->config());
        }
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

  lowered_graph->graph().operands().iterate(
      [](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

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

  auto exec =
      new exec::LinearExecutor{std::move(lowered_graph), tensor_regs, std::move(code_map), order};

  if (!options.trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp =
        std::make_unique<exec::TracingObserver>(options.trace_filepath, exec->graph());
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

exec::IExecutor *ExecutorFactory::createDataflowExecutor(
    std::unique_ptr<compiler::LoweredGraph> lowered_graph, const compiler::CompilerOptions &options,
    const std::shared_ptr<exec::ExecutorMap> &executor_map, bool parallel)
{
  const auto &backend_contexts = lowered_graph->backend_contexts();

  initializeBackendContext(lowered_graph.get());

  auto order = Linear::linearize(*lowered_graph);
  runTensorRegistration(lowered_graph.get(), order);

  initializeSubgraphIOTensors(
      *lowered_graph, (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
                          ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  TensorBuilders tensor_builders{lowered_graph->backend_contexts(), true};
  TensorRegistries tensor_regs{lowered_graph->backend_contexts(), true};

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

  prepareMigrantTensors(*lowered_graph);

  ExecutionBuilder builder;

  // Generate kernels
  lowered_graph->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &op_seq_index,
                                        const ir::OpSequence &op_seq) {
    auto lower_info = lowered_graph->getLowerInfo(op_seq_index);
    auto kernel_gen = lowered_graph->backend_contexts().at(lower_info->backend())->kernel_gen;
    // Set TensorBuilderSet and ExecutorMap to kernel_gen of control flow
    auto cf_kernel_gen = dynamic_cast<backend::controlflow::KernelGenerator *>(kernel_gen.get());
    if (cf_kernel_gen != nullptr)
    {
      assert(cf_kernel_gen != nullptr);
      cf_kernel_gen->setTensorRegistries(tensor_regs);
      cf_kernel_gen->setExecutorMap(executor_map);
    }
    auto fn_seq = kernel_gen->generate(op_seq);
    if (options.he_profiling_mode)
    {
      fn_seq->wrap<SyncFunction>(lower_info->backend()->config());
    }
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

  lowered_graph->graph().operands().iterate(
      [](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

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
    exec = new exec::ParallelExecutor{std::move(lowered_graph), tensor_regs, std::move(code_map)};
  }
  else
  {
    auto dataflow_exec =
        new exec::DataflowExecutor{std::move(lowered_graph), tensor_regs, std::move(code_map)};
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
    std::unique_ptr<exec::IExecutionObserver> ctp =
        std::make_unique<exec::TracingObserver>(options.trace_filepath, exec->graph());
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

} // namespace compiler
} // namespace onert
