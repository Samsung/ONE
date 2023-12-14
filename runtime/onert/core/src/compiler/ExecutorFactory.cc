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

#include "Linear.h"
#include "../backend/builtin/BackendContext.h"
#include "../backend/builtin/Config.h"
#include "../backend/builtin/UserTensor.h"
#include "../dumper/text/GraphDumper.h"
#include "../exec/DataflowExecutor.h"
#include "../exec/ExecTime.h"
#include "../exec/ExecutionObservers.h"
#include "../exec/LinearExecutor.h"
#ifdef MINMAX_H5DUMPER
#include "../exec/MinMaxRecorder.h"
#endif
#include "../exec/ParallelExecutor.h"
#include "../ir/OperationCloner.h"

#include <backend/IPortableTensor.h>
#include <compiler/BackendManager.h>
#include <compiler/ExecutionBuilder.h>
#include <util/TracingCtx.h>

#include <functional>
#include <memory>

#ifdef ONERT_TRAIN
#include <compiler/train/TrainingInfo.h>
#include "../backend/builtin/train/BackendContext.h"
#include "../exec/train/TrainableExecutor.h"

#include <backend/train/TrainableBackendContext.h>
#include <backend/train/ITrainableBackend.h>
#endif // ONERT_TRAIN

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

using DeallocList = std::vector<backend::ITensor *>;
// Deallocation after execution of an operation used by Linear Executor
class DeallocFunction final : public exec::IFunction
{
public:
  DeallocFunction(const DeallocList &tensors) : _dealloc_list{tensors} {}

  void run() override
  {
    for (auto &&tensor : _dealloc_list)
    {
      if (!tensor->is_dynamic())
        continue;
      tensor->deallocBuffer();
    }
  }

private:
  DeallocList _dealloc_list;
};

// TODO Unify initializeSubgraphIOTensors
void initializeSubgraphIOTensors(compiler::ILoweredGraph &lowered_graph,
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

  for (auto &&ind : indices)
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

#ifdef ONERT_TRAIN
void initializeSubgraphIOTensors(compiler::ILoweredGraph &lowered_graph,
                                 const backend::train::TrainableBackendContexts &backend_contexts,
                                 const ir::OperandIndexSequence &indices)
{
  std::shared_ptr<backend::builtin::train::TensorRegistry> builtin_tensor_reg;
  for (const auto &e : backend_contexts)
  {
    auto backend = e.first;
    auto &context = e.second;
    if (backend->config()->id() == backend::builtin::Config::ID)
    {
      builtin_tensor_reg = std::dynamic_pointer_cast<backend::builtin::train::TensorRegistry>(
        context->tensor_registry());
    }
  }
  assert(builtin_tensor_reg);

  for (auto &&ind : indices)
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
#endif // ONERT_TRAIN

backend::BackendContexts
createBackendContexts(compiler::ILoweredGraph &lgraph, bool linear_executor,
                      std::shared_ptr<backend::custom::IKernelBuilder> custom_kernel_builder)
{
  backend::BackendContexts contexts;
  auto &backend_manager = compiler::BackendManager::get();

  std::unordered_map<const backend::Backend *, backend::ContextData> context_data_map;

  // Generate partial graphs for each backend
  for (auto &&backend : backend_manager.getAll())
  {
    auto &data = context_data_map[backend];
    auto graph = std::make_unique<ir::Graph>();
    graph->setLayout(lgraph.graph().layout());
    data.graph = std::move(graph);
  }

  auto &whole_graph = lgraph.graph();
  // Separate operands into partial graphs
  whole_graph.operands().iterate([&](const ir::OperandIndex &operand_ind, ir::Operand &operand) {
    auto &operand_li = lgraph.lower_info().operand;
    const auto &def_factors = operand_li.at(operand_ind).def_factors();
    if (def_factors.size() == 0) // Ignore unused tensor
      return;
    const auto &def_factor = def_factors.getOnlyElement();
    const auto backend = def_factor.backend();
    auto &partial_graph = *context_data_map[backend].graph;
    auto &operand_layouts = context_data_map[backend].operand_layouts;
    assert(operand_layouts.find(operand_ind) == operand_layouts.end());
    operand_layouts[operand_ind] = def_factor.layout();

    // Copy the operand and insert it to the partial graph
    auto new_operand = std::make_unique<ir::Operand>(operand);
    new_operand->clearDefUse();
    operand.releaseData(); // Deref data of LoweredGraph
    auto new_operand_ind = partial_graph.addOperand(operand_ind, std::move(new_operand));
    UNUSED_RELEASE(new_operand_ind);
    assert(new_operand_ind == operand_ind);
  });
  // Separate operations into partial graphs
  whole_graph.operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::IOperation &operation) {
      auto &op_li = lgraph.lower_info().operation;
      auto backend = op_li.at(op_ind).backend();
      auto &partial_graph = *context_data_map[backend].graph;
      auto &external_operands = context_data_map[backend].external_operands;
      auto &operand_layouts = context_data_map[backend].operand_layouts;

      {
        // Add missing operands (externals)
        auto io_list = (operation.getInputs() + operation.getOutputs()) | ir::Remove::DUPLICATED |
                       ir::Remove::UNDEFINED;
        for (auto &&operand_ind : io_list)
        {
          if (partial_graph.operands().exist(operand_ind))
            continue;

          // Copy the operand and insert it to the partial graph
          const auto &operand = whole_graph.operands().at(operand_ind);
          auto new_operand = std::make_unique<ir::Operand>(operand);
          new_operand->clearDefUse();
          auto new_operand_ind = partial_graph.addOperand(operand_ind, std::move(new_operand));
          UNUSED_RELEASE(new_operand_ind);
          assert(new_operand_ind == operand_ind);

          auto layout =
            lgraph.lower_info().operand.at(operand_ind).def_factors().getOnlyElement().layout();
          assert(operand_layouts.find(operand_ind) == operand_layouts.end());
          operand_layouts[operand_ind] = layout;
          external_operands.add(operand_ind);
        }

        auto new_op_ind = partial_graph.addOperation(op_ind, clone(operation));
        UNUSED_RELEASE(new_op_ind);
        assert(new_op_ind == op_ind);
      }
    });

  // Create contexts
  auto whole_op_order = lgraph.graph().topolSortOperations();
  for (auto &&pair : context_data_map)
  {
    auto backend = pair.first;
    auto &data = pair.second;
    // Handle graph input/outputs or external tensors
    data.graph->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &operand) {
      if (whole_graph.getInputs().contains(ind) || whole_graph.getOutputs().contains(ind))
        data.external_operands.add(ind);
      // Inputs are either "graph input" or "no def op and non-constant"
      if (whole_graph.getInputs().contains(ind) ||
          (!operand.getDef().valid() && !operand.isConstant()))
        // Outputs are either "graph output" or "no uses"
        data.graph->addInput(ind);
      if (whole_graph.getOutputs().contains(ind) || operand.getUses().size() == 0)
        data.graph->addOutput(ind);
    });
    VERBOSE(ExecutorFactory) << "createBackendContexts: partial graph for backend="
                             << backend->config()->id() << std::endl;
    dumper::text::dumpGraph(*data.graph);

    std::copy_if(whole_op_order.begin(), whole_op_order.end(), std::back_inserter(data.op_order),
                 [&](const auto &ind) { return data.graph->operations().exist(ind); });
    data.is_linear_executor = linear_executor;
    data.custom_kernel_builder = custom_kernel_builder;
    contexts.emplace(backend, backend->newContext(std::move(data)));
  }
  return contexts;
}

template <typename Context>
std::deque<std::pair<const backend::Backend *, Context *>> orderBackendContext(
  const std::unordered_map<const backend::Backend *, std::unique_ptr<Context>> &tbackend_contexts)
{
  std::deque<std::pair<const backend::Backend *, Context *>> ordered_contexts;

  for (auto &&pair : tbackend_contexts)
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

  return ordered_contexts;
}

} // namespace
} // namespace onert

namespace onert
{
namespace compiler
{
void prepareMigrantTensors(compiler::ILoweredGraph &lowered_graph,
                           const backend::BackendContexts &backend_contexts)
{
  TensorRegistries tensor_regs{backend_contexts, true};

  lowered_graph.graph().operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::IOperation &op) {
      auto lower_info = lowered_graph.lower_info().operation.getRawPtr(op_ind);
      auto &backend_ctx = backend_contexts.at(lower_info->backend());
      for (auto &&ind :
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

void prepareBuiltinBackend(const TensorRegistries &tensor_regs,
                           const std::shared_ptr<exec::IExecutors> &executors,
                           const backend::BackendContexts &backend_contexts,
                           const ir::ModelIndex &index)
{
  for (auto &&pair : backend_contexts)
  {
    auto builtin_context = dynamic_cast<backend::builtin::BackendContext *>(pair.second.get());
    if (builtin_context != nullptr)
    {
      auto builtin_kernel_gen = builtin_context->kernel_gen;
      builtin_kernel_gen->setTensorRegistries(tensor_regs);
      builtin_kernel_gen->setExecutors(executors);
      builtin_kernel_gen->setModelIndex(index);
    }
  }
}

std::deque<std::pair<const backend::Backend *, backend::BackendContext *>>
orderBackendContext(const backend::BackendContexts &backend_contexts)
{
  std::deque<std::pair<const backend::Backend *, backend::BackendContext *>> ordered_contexts;
  for (auto &&pair : backend_contexts)
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
  return ordered_contexts;
}

exec::IExecutor *createLinearExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                                      const std::shared_ptr<exec::IExecutors> &executors,
                                      const ExecutorFactoryArgs &args)
{
  const auto options = args.options;
  const auto &model_index = args.model_index;
  const auto tracing_ctx = args.tracing_ctx;
  auto custom_kernel_builder = args.custom_kernel_builder;
  auto &graph = lowered_graph->graph();

  backend::BackendContexts backend_contexts =
    createBackendContexts(*lowered_graph, options->executor == "Linear", custom_kernel_builder);

  TensorRegistries tensor_regs{backend_contexts, true};

  initializeSubgraphIOTensors(
    *lowered_graph, backend_contexts,
    (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
      ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  // linearize
  auto order = Linear::linearize(*lowered_graph);
  Linear::dump(*lowered_graph, order);

  for (auto &&pair : backend_contexts)
  {
    pair.second->genTensors();
  }

  prepareMigrantTensors(*lowered_graph, backend_contexts);

  // Give some runtime objects to builtin KernelGenerator
  prepareBuiltinBackend(tensor_regs, executors, backend_contexts, model_index);

  ExecutionBuilder builder;

  // Adjust the order of backends for the upcoming iteration
  auto ordered_contexts = orderBackendContext(backend_contexts);

  // Simulate the execution for deallocation of tensors
  std::unordered_map<ir::OperationIndex, DeallocList> dealloc_list_map;
  {
    ir::OperandIndexMap<uint32_t> uses_map;
    ir::OperandIndexSequence constants;

    auto model_io =
      (graph.getInputs() + graph.getOutputs()) | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;

    // Prepare scanning
    graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
      uses_map[ind] = obj.getUses().size();

      if (obj.isConstant())
        constants.append(ind);
    });

    // A trick to consider constants as an execption
    for (const auto &ind : constants)
    {
      uses_map[ind]++;
    }

    for (const auto &op_ind : order)
    {
      const auto &op = graph.operations().at(op_ind);
      auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
      auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

      for (const auto &ind : op_inputs)
      {
        const auto &operand = graph.operands().at(ind);
        assert(uses_map.find(ind) != uses_map.end());
        assert(uses_map[ind] > 0);
        uses_map[ind]--;
        if (uses_map[ind] == 0 && !operand.info().isVariable() && !model_io.contains(ind))
        {
          dealloc_list_map[op_ind].emplace_back(tensor_regs.getITensor(ind));
        }
      }
    }

    // Dispose and validate
    for (const auto &ind : constants)
    {
      --uses_map[ind];
    }

    assert(
      std::all_of(uses_map.begin(), uses_map.end(),
                  [](std::pair<const ir::OperandIndex, uint32_t> it) { return it.second == 0; }));
  }

  // Generate kernels
  for (auto &&pair : ordered_contexts)
  {
    auto codes = pair.second->genKernels();
    for (auto &&pair : codes)
    {
      auto &op_ind = pair.first;
      auto &fn_seq = pair.second;
      auto &op = lowered_graph->graph().operations().at(op_ind);
      auto lower_info = lowered_graph->lower_info().operation.getRawPtr(op_ind);
      if (options->he_profiling_mode)
        fn_seq->wrap<SyncFunction>(lower_info->backend()->config());
      if (!dealloc_list_map[op_ind].empty())
        fn_seq->append(std::make_unique<DeallocFunction>(dealloc_list_map[op_ind]));
      builder.append(op_ind, {op_ind, &op, lower_info, std::move(fn_seq)});
    }
  }

  auto code_map = builder.releaseCodeMap();

  auto exec = new exec::LinearExecutor{std::move(lowered_graph),
                                       std::move(backend_contexts),
                                       tensor_regs,
                                       std::move(code_map),
                                       order,
                                       tracing_ctx};

  if (!options->trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp =
      std::make_unique<exec::TracingObserver>(options->trace_filepath, exec->graph(), tracing_ctx);
    exec->addObserver(std::move(ctp));
  }
#ifdef MINMAX_H5DUMPER
  if (!options->minmax_filepath.empty())
    exec->addObserver(std::make_unique<exec::MinMaxRecorder>(
      options->minmax_filepath, exec->graph(), exec->getBackendContexts()));
#endif

  return exec;
}

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
                                         const std::shared_ptr<exec::IExecutors> &executors,
                                         const ExecutorFactoryArgs &args)
{
  assert(args.options != nullptr);
  return _map.at(args.options->executor)(std::move(lowered_graph), executors, args);
}

exec::IExecutor *
ExecutorFactory::createDataflowExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                                        const std::shared_ptr<exec::IExecutors> &executors,
                                        const ExecutorFactoryArgs &args, bool parallel)
{
  const auto options = args.options;
  const auto &model_index = args.model_index;
  const auto tracing_ctx = args.tracing_ctx;
  auto custom_kernel_builder = args.custom_kernel_builder;

  backend::BackendContexts backend_contexts =
    createBackendContexts(*lowered_graph, options->executor == "Linear", custom_kernel_builder);

  TensorRegistries tensor_regs{backend_contexts, true};

  initializeSubgraphIOTensors(
    *lowered_graph, backend_contexts,
    (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
      ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  for (auto &&pair : backend_contexts)
  {
    pair.second->genTensors();
  }

  prepareMigrantTensors(*lowered_graph, backend_contexts);

  // Give some runtime objects to builtin KernelGenerator
  prepareBuiltinBackend(tensor_regs, executors, backend_contexts, model_index);

  ExecutionBuilder builder;

  // Adjust the order of backends for the upcoming iteration
  auto ordered_contexts = orderBackendContext(backend_contexts);

  // Generate kernels
  for (auto &&pair : ordered_contexts)
  {
    auto codes = pair.second->genKernels();
    for (auto &&pair : codes)
    {
      auto &op_ind = pair.first;
      auto &fn_seq = pair.second;
      auto &op = lowered_graph->graph().operations().at(op_ind);
      auto lower_info = lowered_graph->lower_info().operation.getRawPtr(op_ind);
      if (options->he_profiling_mode)
        fn_seq->wrap<SyncFunction>(lower_info->backend()->config());
      builder.append(op_ind, {op_ind, &op, lower_info, std::move(fn_seq)});
    }
  }

  auto code_map = builder.releaseCodeMap();

  exec::ExecutorBase *exec = nullptr;
  if (parallel)
  {
    exec = new exec::ParallelExecutor{std::move(lowered_graph), std::move(backend_contexts),
                                      tensor_regs, std::move(code_map), tracing_ctx};
  }
  else
  {
    auto dataflow_exec =
      new exec::DataflowExecutor{std::move(lowered_graph), std::move(backend_contexts), tensor_regs,
                                 std::move(code_map), tracing_ctx};
    if (options->he_profiling_mode)
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

  if (!options->trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp =
      std::make_unique<exec::TracingObserver>(options->trace_filepath, exec->graph(), tracing_ctx);
    exec->addObserver(std::move(ctp));
  }

  return exec;
}

#ifdef ONERT_TRAIN
void prepareMigrantTensors(compiler::ILoweredGraph &lowered_graph,
                           const backend::train::TrainableBackendContexts &backend_contexts)
{
  train::TensorRegistries tensor_regs{backend_contexts, true};

  lowered_graph.graph().operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::IOperation &op) {
      auto lower_info = lowered_graph.lower_info().operation.getRawPtr(op_ind);
      auto &backend_ctx = backend_contexts.at(lower_info->backend());
      for (auto &&ind :
           (op.getInputs() + op.getOutputs()) | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
      {
        // If an Operation's input/output tensor does not have an own tensor object,
        // it must be using migrant tensors, so find the tensor from other tensor registries and
        // register it to the current tensor registry if it is portable
        if (!backend_ctx->tensor_registry()->getITensor(ind))
        {
          auto tensor = tensor_regs.getITensor(ind);
          assert(tensor); // The tensor must have been registered
          auto ptensor = dynamic_cast<backend::IPortableTensor *>(tensor);
          if (ptensor)
            backend_ctx->tensor_registry()->setMigrantTensor(ind, ptensor);
        }
      }
    });
}

exec::IExecutor *
createTrainableExecutor(std::unique_ptr<compiler::train::LoweredTrainableGraph> lowered_graph,
                        const std::shared_ptr<exec::IExecutors> &, const ExecutorFactoryArgs &args,
                        const compiler::train::TrainingInfo &training_info)
{
  const auto options = args.options;
  const auto tracing_ctx = args.tracing_ctx;
  auto custom_kernel_builder = args.custom_kernel_builder;

  auto &graph = lowered_graph->graph();

  lowered_graph->trainable_graph().operations().iterate([](const onert::ir::OperationIndex &,
                                                           const onert::ir::IOperation &op) {
    try
    {
      UNUSED_RELEASE(dynamic_cast<const ir::train::ITrainableOperation &>(op));
    }
    catch (std::bad_cast &)
    {
      throw std::runtime_error("ExecutorFactory: " + op.name() + " is not trainable operation yet");
    }
  });

  // TODO Create context only once instead of replacing
  backend::train::TrainableBackendContexts tbackend_contexts;
  backend::BackendContexts base_backend_contexts =
    createBackendContexts(*lowered_graph, true, custom_kernel_builder);

  // Replace BackendContext with TrainbleBackendContext
  for (auto &&pair : base_backend_contexts)
  {
    auto ctx = pair.second.get();
    const auto &data = ctx->data();

    // Create partial and trainable graphs
    auto tgraph = std::make_unique<ir::train::TrainableGraph>(*data.graph);
    data.graph->operations().iterate(
      [&](const onert::ir::OperationIndex &op_index, const onert::ir::IOperation &) {
        const auto &orig_tgraph = lowered_graph->trainable_graph();
        const auto &trainable_op = orig_tgraph.operation(op_index);
        auto gen_index = tgraph->replaceOperation(op_index, trainable_op.clone());
        UNUSED_RELEASE(gen_index);
        assert(gen_index == op_index);
      });
    data.graph->operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &) {
      const auto &orig_tgraph = lowered_graph->trainable_graph();
      if (orig_tgraph.back_props().exist(index))
      {
        const auto &back_prop = orig_tgraph.back_props().at(index);
        auto new_back_prop = std::make_unique<ir::Operand>(back_prop);
        auto gen_index = tgraph->addBackProp(index, std::move(new_back_prop));
        UNUSED_RELEASE(gen_index);
        assert(gen_index == index);
      }
    });

    // Remove outputs of whole graph from external_operands
    auto external_operands = data.external_operands;
    for (const auto &index : lowered_graph->trainable_graph().getOutputs())
    {
      if (external_operands.contains(index))
        external_operands.remove(index);
    }

    // Set trainable context data
    backend::train::TrainableContextData tdata;
    tdata.tgraph = std::move(tgraph);
    tdata.op_order = std::move(data.op_order);
    tdata.external_operands = std::move(external_operands);
    tdata.operand_layouts = std::move(data.operand_layouts);
    tdata.custom_kernel_builder = std::move(data.custom_kernel_builder);
    tdata.is_linear_executor = data.is_linear_executor;
    tdata.optim_info = training_info.optimizerInfo();

    // TODO Remove dynamic_cast
    try
    {
      const auto backend = pair.first;
      const auto tbackend = dynamic_cast<const backend::train::ITrainableBackend *>(backend);
      tbackend_contexts.emplace(backend, tbackend->newContext(std::move(tdata)));
    }
    catch (const std::bad_cast &)
    {
      throw std::runtime_error("ExecutorFactory: Invalid backend - TrainableExecutor does not "
                               "support non-trainble backends");
    }
  }
  base_backend_contexts.clear();

  train::TensorRegistries tensor_regs{tbackend_contexts, true};

  initializeSubgraphIOTensors(
    *lowered_graph, tbackend_contexts,
    (lowered_graph->graph().getInputs() + lowered_graph->graph().getOutputs()) |
      ir::Remove::DUPLICATED | ir::Remove::UNDEFINED);

  // linearize for forwarding
  auto order = Linear::linearize(*lowered_graph);
  VERBOSE(ExecutorFactory) << "Linearize for forwarding order" << std::endl;
  Linear::dump(*lowered_graph, order);

  // linearize for backwarding
  auto backward_order = lowered_graph->trainable_graph().btopolSortOperations();
  VERBOSE(ExecutorFactory) << "Linearize for backwarding order" << std::endl;
  Linear::dump(*lowered_graph, backward_order);

  for (auto &&pair : tbackend_contexts)
  {
    pair.second->genTensors();
  }

  for (auto &&pair : tbackend_contexts)
  {
    auto tctx = pair.second.get();
    tctx->genTrainingTensors();
  }

  prepareMigrantTensors(*lowered_graph, tbackend_contexts);

  // Give some runtime objects to builtin KernelGenerator
  for (auto &&pair : tbackend_contexts)
  {
    auto builtin_context =
      dynamic_cast<backend::builtin::train::BackendContext *>(pair.second.get());
    if (builtin_context != nullptr)
    {
      auto builtin_kernel_gen = builtin_context->kernel_gen;
      builtin_kernel_gen->setTensorRegistries(tensor_regs);
      builtin_kernel_gen->setWholeGraphOutputs(lowered_graph->trainable_graph().getOutputs());
    }
  }

  // Adjust the order of backends for the upcoming iteration
  auto ordered_contexts =
    onert::orderBackendContext<backend::train::TrainableBackendContext>(tbackend_contexts);

  // TODO Remove this simulation
  // Simulate the execution for deallocation of tensors
  std::unordered_map<ir::OperationIndex, DeallocList> dealloc_list_map;
  {
    ir::OperandIndexMap<uint32_t> uses_map;
    ir::OperandIndexSequence constants;

    auto model_io =
      (graph.getInputs() + graph.getOutputs()) | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;

    // Prepare scanning
    graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
      uses_map[ind] = obj.getUses().size();

      if (obj.isConstant())
        constants.append(ind);
    });

    // A trick to consider constants as an execption
    for (const auto &ind : constants)
    {
      uses_map[ind]++;
    }

    for (const auto &op_ind : order)
    {
      const auto &op = graph.operations().at(op_ind);
      auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
      auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

      for (const auto &ind : op_inputs)
      {
        const auto &operand = graph.operands().at(ind);
        assert(uses_map.find(ind) != uses_map.end());
        assert(uses_map[ind] > 0);
        uses_map[ind]--;
        if (uses_map[ind] == 0 && !operand.info().isVariable() && !model_io.contains(ind))
        {
          dealloc_list_map[op_ind].emplace_back(tensor_regs.getITensor(ind));
        }
      }
    }

    // Dispose and validate
    for (const auto &ind : constants)
    {
      --uses_map[ind];
    }

    assert(
      std::all_of(uses_map.begin(), uses_map.end(),
                  [](std::pair<const ir::OperandIndex, uint32_t> it) { return it.second == 0; }));
  }

  // Check back propagation tensors
  {
    // TODO Support multiple subgraphs
    // Check if the back propagation tensors corresponding to inputs of model are nullptr
    // NOTE The back propagation tensors corresponding to inputs of model are for inputs of
    // PermuteLayers
    //      and they are nullptr and because they are meaningless.
    assert(std::all_of(
      lowered_graph->trainable_graph().getInputs().begin(),
      lowered_graph->trainable_graph().getInputs().end(),
      [&](const auto &input_idx) { return tensor_regs.getBackPropITensor(input_idx) == nullptr; }));

    // Check if the back propagation tensors corresponding to outputs of model exist
    assert(std::all_of(lowered_graph->trainable_graph().getOutputs().begin(),
                       lowered_graph->trainable_graph().getOutputs().end(),
                       [&](const auto &output_idx) {
                         return tensor_regs.getBackPropITensor(output_idx) == nullptr;
                       }));
  }

  train::TrainableCodeMap code_map;
  // Generate kernels
  for (auto &&pair : ordered_contexts)
  {
    auto codes = pair.second->genKernels();
    for (auto &&pair : codes)
    {
      auto &op_ind = pair.first;
      auto &tn_seq = pair.second;
      auto &op = lowered_graph->trainable_graph().operation(op_ind);
      auto lower_info = lowered_graph->lower_info().operation.getRawPtr(op_ind);

      assert(code_map.find(op_ind) == code_map.end());
      code_map.insert(
        {op_ind, train::TrainableCodeAndInfo{op_ind, &op, lower_info, std::move(tn_seq)}});
    }
  }

  if (order.size() != code_map.size())
  {
    throw std::runtime_error("ExecutorFactory: Some kernels are not generated");
  }

  auto exec = new exec::train::TrainableExecutor{std::move(lowered_graph),
                                                 std::move(tbackend_contexts),
                                                 tensor_regs,
                                                 std::move(code_map),
                                                 order,
                                                 backward_order,
                                                 tracing_ctx,
                                                 training_info.lossInfo()};

  if (!options->trace_filepath.empty())
  {
    std::unique_ptr<exec::IExecutionObserver> ctp =
      std::make_unique<exec::TracingObserver>(options->trace_filepath, exec->graph(), tracing_ctx);
    exec->addObserver(std::move(ctp));
  }
  // TODO Support MINMAX_H5DUMPER

  return exec;
}

exec::IExecutor *
ExecutorFactory::create(std::unique_ptr<compiler::train::LoweredTrainableGraph> lowered_graph,
                        const std::shared_ptr<exec::IExecutors> &executors,
                        const ExecutorFactoryArgs &args,
                        const compiler::train::TrainingInfo &training_info)
{
  assert(args.options != nullptr);

  if (args.options->executor != "Linear")
    throw std::runtime_error("ExecutorFactory: TrainableExecutor supports only 'Linear' now");

  return createTrainableExecutor(std::move(lowered_graph), executors, args, training_info);
}
#endif // ONERT_TRAIN

} // namespace compiler
} // namespace onert
