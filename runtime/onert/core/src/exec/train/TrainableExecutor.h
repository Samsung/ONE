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

  void execute(const std::vector<backend::IPortableTensor *> &inputs,
               const std::vector<backend::IPortableTensor *> &outputs,
               const ExecutionOptions &options) override
  {
    forward(inputs, outputs, options, false);
  }

  uint32_t inputSize() const override { return _input_tensors.size(); }

  uint32_t outputSize() const override { return _output_tensors.size(); }

  const ir::OperandInfo &inputInfo(uint32_t index) const override
  {
    return _input_tensors[index]->get_info();
  }

  const ir::OperandInfo &outputInfo(uint32_t index) const override
  {
    return _output_tensors[index]->get_info();
  }

  ir::Layout inputLayout(uint32_t index) const override { return _input_tensors[index]->layout(); }

  ir::Layout outputLayout(uint32_t index) const override
  {
    return _output_tensors[index]->layout();
  }

  void forward(const std::vector<backend::IPortableTensor *> &inputs,
               const std::vector<backend::IPortableTensor *> &outputs,
               const ExecutionOptions &options, bool training);
  void backward(const ExecutionOptions &options, uint32_t training_step);

  // Used only in Dataflow and Parallel Executors
  void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>> ranks) final
  {
    _indexed_ranks = std::move(ranks);
  };

  void addObserver(std::unique_ptr<IExecutionObserver> ref) { _observers.add(std::move(ref)); };

  float getLoss(const ir::IOIndex &pred_io_ind) const;

  void iterateTrainableTensors(
    const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)>
      &fn) const;

  backend::train::TrainableBackendContexts &getBackendContexts() { return _backend_contexts; }

  const ExecutionOptions &currentOptions() const override { return _current_options; }

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
  /**
   * It is set by execute() method only in thread-safe environment.
   * It is used for non-primary executor call on builtin backend
   * and accessed by entryExecutor's currentOptions() method.
   *
   * TODO: Find better way to pass config to non-primary executor
   */
  ExecutionOptions _current_options;
};

} // namespace train
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_TRAINABLE_EXECUTOR_H_
