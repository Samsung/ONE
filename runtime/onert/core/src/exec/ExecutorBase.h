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

#ifndef __ONERT_EXEC_EXECUTOR_BASE_H__
#define __ONERT_EXEC_EXECUTOR_BASE_H__

#include "ExecutionObservee.h"
#include "../backend/builtin/IOTensor.h"
#include "../compiler/TensorRegistries.h"

#include "compiler/LoweredGraph.h"
#include "exec/IExecutor.h"
#include "exec/IODescription.h"
#include "ir/Graph.h"
#include "ir/OperationIndexMap.h"
#include "util/TracingCtx.h"

#include <memory>
#include <mutex>
#include <vector>

namespace onert
{
namespace exec
{

class ExecutorBase : public IExecutor
{
public:
  /**
   * @brief Construct a new ExecutorBase object
   * @param graph Graph object
   * @param tensor_builders Tensor builders that are currently used
   */
  ExecutorBase(std::unique_ptr<compiler::LoweredGraph> &&lowered_graph,
               backend::BackendContexts &&backend_contexts,
               const compiler::TensorRegistries &tensor_regs, const util::TracingCtx *tracing_ctx);

  virtual ~ExecutorBase() = default;

  const ir::Graph &graph() final { return _graph; }

  const ir::Graph &parent_graph() final { return _parent_graph; }

  void execute(const IODescription &desc) final;

  void execute(const std::vector<backend::IPortableTensor *> &inputs,
               const std::vector<backend::IPortableTensor *> &outputs) override;

  // Used only in Dataflow and Parallel Executors
  void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>> ranks) final
  {
    _indexed_ranks = std::move(ranks);
  };

  virtual void executeImpl(void) = 0;

  void addObserver(std::unique_ptr<IExecutionObserver> ref) { _subject.add(std::move(ref)); };

  const std::vector<backend::builtin::IOTensor *> &getOutputTensors() const override
  {
    return _output_tensors;
  }

protected:
  /**
   * @brief Returns @c true if any input tensor is dynamic; @c false if all are static tensors
   */
  bool hasDynamicInput();

protected:
  ExecutionObservee _subject;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  std::unique_ptr<compiler::LoweredGraph> _lowered_graph;
  backend::BackendContexts _backend_contexts;
  const ir::Graph &_graph;
  const ir::Graph &_parent_graph;
  std::vector<backend::builtin::IOTensor *> _input_tensors;
  std::vector<backend::builtin::IOTensor *> _output_tensors;
  std::mutex _mutex;
  const util::TracingCtx *_tracing_ctx;

private:
  void handleDynamicInputTensor(ir::IOIndex input_index, const IODescription &desc);
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTOR_BASE_H__
