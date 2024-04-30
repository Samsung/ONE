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

#ifndef __ONERT_BACKEND_BACKEND_TRAIN_TRAINABLE_CONTEXT_H__
#define __ONERT_BACKEND_BACKEND_TRAIN_TRAINABLE_CONTEXT_H__

#include "backend/Backend.h"
#include "backend/train/ITensorRegistry.h"
#include "backend/train/ITrainableBackend.h"
#include "exec/train/TrainableFnSequence.h"
#include "ir/OperandIndexMap.h"
#include "ir/train/OptimizerInfo.h"
#include "ir/train/TrainableGraph.h"
#include "util/Set.h"

namespace onert
{
namespace backend
{
namespace train
{

using FunctionMap =
  std::unordered_map<ir::OperationIndex, std::unique_ptr<exec::train::TrainableFnSequence>>;

struct TrainableContextData
{
  // A partial and trainable graph that only includes used operand/operations of the original graph
  std::unique_ptr<ir::train::TrainableGraph> tgraph;
  /* A linear order of operations. This is neccessary for when a graph is not fully connected */
  std::vector<onert::ir::OperationIndex> op_order;
  /* Operands that are defined by other backends */
  util::Set<ir::OperandIndex> external_operands;
  /* Operand layout info */
  ir::OperandIndexMap<ir::Layout> operand_layouts;
  /* Custom kernel builder */
  std::shared_ptr<custom::IKernelBuilder> custom_kernel_builder;
  /* Is linear executor or not */
  bool is_linear_executor;
  /* Optimizer information */
  ir::train::OptimizerInfo optim_info;
};

class TrainableBackendContext
{
public:
  TrainableBackendContext(const ITrainableBackend *backend,
                          std::unique_ptr<TrainableContextData> &&tdata,
                          std::shared_ptr<ITensorRegistry> tensor_registry = nullptr)
    : _backend{backend}, _tdata{std::move(tdata)}, _tensor_registry{tensor_registry}
  {
    assert(_tdata);
  }
  virtual ~TrainableBackendContext() = default;

  const ir::train::TrainableGraph *trainable_graph() const { return _tdata->tgraph.get(); }

  const TrainableContextData *data() const { return _tdata.get(); }

  const ITrainableBackend *backend() const { return _backend; }
  const util::Set<ir::OperandIndex> &external_operands() const { return _tdata->external_operands; }
  const ir::OperandIndexMap<ir::Layout> &operand_layouts() const { return _tdata->operand_layouts; }

  std::shared_ptr<ITensorRegistry> tensor_registry() { return _tensor_registry; }

  virtual ITensorRegistry *genTrainingTensors() = 0;
  virtual backend::ITensorRegistry *genTensors() = 0;
  virtual FunctionMap genKernels() = 0;

private:
  const ITrainableBackend *_backend{nullptr};

protected:
  std::unique_ptr<TrainableContextData> _tdata;

protected:
  std::shared_ptr<ITensorRegistry> _tensor_registry;
};

using TrainableBackendContexts =
  std::unordered_map<const Backend *, std::unique_ptr<TrainableBackendContext>>;

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BACKEND_TRAIN_TRAINABLE_CONTEXT_H__
