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

#ifndef __ONERT_EXEC_TRAIN_TRAINABLE_EXECUTOR_H_
#define __ONERT_EXEC_TRAIN_TRAINABLE_EXECUTOR_H_

#include "exec/IExecutor.h"

#include "../ExecutionObservee.h"
#include "../../compiler/train/TensorRegistries.h"

#include "backend/train/TrainableBackendContext.h"
#include "compiler/train/TrainableCodeMap.h"
#include "compiler/train/LoweredTrainableGraph.h"
#include "ir/train/LossInfo.h"
#include "ir/Index.h"
#include "util/TracingCtx.h"

namespace onert
{
namespace exec
{
namespace train
{

class TrainableExecutor : public IExecutor
{
public:
  /**
   * @brief Construct a new TrainableExecutor object
   * @param lowered_graph LoweredTrainableGraph object
   * @param tensor_builders Tensor builders that are currently used
   * @param code_map @c ir::Operation and its code map
   */
  TrainableExecutor(std::unique_ptr<compiler::train::LoweredTrainableGraph> lowered_graph,
                    backend::train::TrainableBackendContexts &&backend_contexts,
                    const compiler::train::TensorRegistries &tensor_regs,
                    compiler::train::TrainableCodeMap &&code_map,
                    const std::vector<ir::OperationIndex> &forward_order,
                    const std::vector<ir::OperationIndex> &backward_order,
                    const util::TracingCtx *tracing_ctx, const ir::train::LossInfo &training_info);

public:
  const ir::Graph &graph() const final { return _trainable_graph.graph(); }

  void execute(const ExecutionContext &ctx) override { forward(ctx, false); };

  void execute(const std::vector<backend::IPortableTensor *> &inputs,
               const std::vector<backend::IPortableTensor *> &outputs) override;

  void forward(const ExecutionContext &ctx, bool training);
  void backward(const ExecutionContext &ctx, uint32_t training_step);

  // Used only in Dataflow and Parallel Executors
  void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>> ranks) final
  {
    _indexed_ranks = std::move(ranks);
  };

  void addObserver(std::unique_ptr<IExecutionObserver> ref) { _observers.add(std::move(ref)); };

  const std::vector<backend::builtin::IOTensor *> &getInputTensors() const override
  {
    return _input_tensors;
  }

  const std::vector<backend::builtin::IOTensor *> &getOutputTensors() const override
  {
    return _output_tensors;
  }

  float getLoss(const ir::IOIndex &pred_io_ind) const;

  void iterateTrainableTensors(
    const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)>
      &fn) const;

  backend::train::TrainableBackendContexts &getBackendContexts() { return _backend_contexts; }

private:
  void forwardImpl(const ExecutionObservee &subject, bool training);
  void backwardImpl(const ExecutionObservee &subject, uint32_t training_step);

private:
  compiler::train::TrainableCodeMap _code_map;
  std::vector<ir::OperationIndex> _forward_order;
  std::vector<ir::OperationIndex> _backward_order;
  ExecObservers _observers;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  std::unique_ptr<compiler::train::LoweredTrainableGraph> _lowered_graph;
  backend::train::TrainableBackendContexts _backend_contexts;
  const ir::train::TrainableGraph &_trainable_graph;
  compiler::train::TensorRegistries _tensor_regs;
  std::vector<backend::builtin::IOTensor *> _input_tensors;
  std::vector<backend::builtin::IOTensor *> _output_tensors;
  std::mutex _mutex;
  const util::TracingCtx *_tracing_ctx;
  const ir::train::LossInfo _loss_info;
};

} // namespace train
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_TRAINABLE_EXECUTOR_H_
