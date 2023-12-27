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

#ifndef __ONERT_COMPILER_EXECUTOR_FACTORY_H__
#define __ONERT_COMPILER_EXECUTOR_FACTORY_H__

#include "TensorRegistries.h"

#include "backend/ITensor.h"
#include "backend/train/TrainableBackendContext.h"
#include "compiler/LoweredGraph.h"
#include "compiler/train/LoweredTrainableGraph.h"
#include "exec/IExecutors.h"
#include "ir/train/TrainingInfo.h"

#include <deque>
#include <unordered_map>

namespace onert
{
namespace compiler
{

// TODO Change to a better name
struct ExecutorFactoryArgs
{
  const util::TracingCtx *tracing_ctx;
  const compiler::CompilerOptions *options;
  ir::ModelIndex model_index;
  std::shared_ptr<backend::custom::IKernelBuilder> custom_kernel_builder;
};

class ExecutorFactory
{
public:
  static ExecutorFactory &get();

public:
  exec::IExecutor *create(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                          const std::shared_ptr<exec::IExecutors> &executors,
                          const ExecutorFactoryArgs &args);

  // TODO Unify create()
  exec::IExecutor *create(std::unique_ptr<compiler::train::LoweredTrainableGraph> lowered_graph,
                          const std::shared_ptr<exec::IExecutors> &executors,
                          const ExecutorFactoryArgs &args,
                          const ir::train::TrainingInfo &training_info);

private:
  ExecutorFactory();

private:
  static void prepareMigrantTensors(compiler::ILoweredGraph &lowered_graph,
                                    const backend::BackendContexts &backend_contexts);
  static void prepareBuiltinBackend(const TensorRegistries &tensor_regs,
                                    const std::shared_ptr<exec::IExecutors> &executors,
                                    const backend::BackendContexts &backend_contexts,
                                    const ir::ModelIndex &index);
  static std::deque<std::pair<const backend::Backend *, backend::BackendContext *>>
  orderBackendContext(const backend::BackendContexts &backend_contexts);

  static exec::IExecutor *
  createLinearExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                       const std::shared_ptr<exec::IExecutors> &executors,
                       const ExecutorFactoryArgs &args);
  static exec::IExecutor *
  createDataflowExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                         const std::shared_ptr<exec::IExecutors> &executors,
                         const ExecutorFactoryArgs &args, bool parallel);
  // TODO Unify prepareMigrantTensors
  static void
  prepareMigrantTensors(compiler::ILoweredGraph &lowered_graph,
                        const backend::train::TrainableBackendContexts &backend_contexts);
  static exec::IExecutor *
  createTrainableExecutor(std::unique_ptr<compiler::train::LoweredTrainableGraph> lowered_graph,
                          const std::shared_ptr<exec::IExecutors> &executors,
                          const ExecutorFactoryArgs &args,
                          const ir::train::TrainingInfo &training_info);

private:
  std::unordered_map<
    std::string, std::function<exec::IExecutor *(std::unique_ptr<compiler::LoweredGraph>,
                                                 const std::shared_ptr<exec::IExecutors> &executors,
                                                 const ExecutorFactoryArgs &args)>>
    _map;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_EXECUTOR_FACTORY_H__
