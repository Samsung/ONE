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
#include "compiler/LoweredGraph.h"
#include "exec/Executors.h"

#include <deque>
#include <unordered_map>

namespace onert
{
namespace compiler
{

class ExecutorFactory
{
public:
  static ExecutorFactory &get();

public:
  exec::IExecutor *create(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                          const util::TracingCtx *tracing_ctx,
                          const compiler::CompilerOptions &options,
                          const std::shared_ptr<exec::Executors> &executors);

  exec::IExecutor *create(std::shared_ptr<const compiler::LoweredGraph> lowered_graph,
                          const util::TracingCtx *tracing_ctx,
                          const compiler::CompilerOptions &options,
                          const std::shared_ptr<exec::Executors> &executors, int32_t batch_num);

private:
  ExecutorFactory();

private:
  static void prepareMigrantTensors(const compiler::LoweredGraph &lowered_graph,
                                    const backend::BackendContexts &backend_contexts);
  static void prepareBuiltinBackend(const TensorRegistries &tensor_regs,
                                    const std::shared_ptr<exec::Executors> &executors,
                                    const backend::BackendContexts &backend_contexts);
  static std::deque<std::pair<const backend::Backend *, backend::BackendContext *>>
  orderBackendContext(const backend::BackendContexts &backend_contexts);

  static exec::IExecutor *
  createLinearExecutor(std::shared_ptr<const compiler::LoweredGraph> lowered_graph,
                       const util::TracingCtx *tracing_ctx,
                       const compiler::CompilerOptions &options,
                       const std::shared_ptr<exec::Executors> &executors, int32_t batch_num);
  static exec::IExecutor *createDataflowExecutor(
    std::shared_ptr<const compiler::LoweredGraph> lowered_graph,
    const util::TracingCtx *tracing_ctx, const compiler::CompilerOptions &options,
    const std::shared_ptr<exec::Executors> &executors, int32_t batch_num, bool parallel);

private:
  std::unordered_map<std::string,
                     std::function<exec::IExecutor *(
                       std::unique_ptr<compiler::LoweredGraph>, const util::TracingCtx *tracing_ctx,
                       const compiler::CompilerOptions &options,
                       const std::shared_ptr<exec::Executors> &executors, int32_t batch_num)>>
    _map;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_EXECUTOR_FACTORY_H__
