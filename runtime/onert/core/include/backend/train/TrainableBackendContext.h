/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "backend/BackendContext.h"
#include "ir/train/TrainableGraph.h"

namespace onert
{
namespace backend
{
namespace train
{

class TrainableContextData : public ContextData
{
public:
  TrainableContextData(void) = delete;
  TrainableContextData(backend::ContextData &&ctx_data, const ir::train::TrainableGraph &tgraph)
    : ContextData{std::move(ctx_data)}
  {
    _tgraph = std::make_unique<ir::train::TrainableGraph>(*graph, tgraph.trainable_operations());
  }
  TrainableContextData(const TrainableContextData &tdata) = delete;
  TrainableContextData(TrainableContextData &&tdata) = default;
  TrainableContextData &operator=(const TrainableContextData &tdata) = delete;
  TrainableContextData &operator=(TrainableContextData &&tdata) = default;

  const ir::train::TrainableGraph *trainable_graph() const { return _tgraph.get(); }

private:
  // A whole trainable (sub)graph that has operations equivalent to the graph in ContextData
  std::unique_ptr<ir::train::TrainableGraph> _tgraph;
};

class TrainableBackendContext : public backend::BackendContext
{
public:
  TrainableBackendContext(const backend::Backend *backend,
                          std::unique_ptr<TrainableContextData> &&tdata,
                          std::shared_ptr<backend::ITensorRegistry> tensor_registry = nullptr,
                          std::shared_ptr<backend::ITensorRegistry> grad_tensor_registry = nullptr)
    : backend::BackendContext{backend, tensor_registry}, _tdata{std::move(tdata)},
      grad_tensor_registry{grad_tensor_registry}
  {
  }

  // TODO Remove this constructor
  TrainableBackendContext(const Backend *backend, ContextData &&data,
                          std::shared_ptr<ITensorRegistry> tensor_registry = nullptr)
    : backend::BackendContext{backend, std::move(data), tensor_registry}, _tdata{nullptr},
      grad_tensor_registry{nullptr}
  {
  }

  virtual ~TrainableBackendContext() = default;

  const ir::train::TrainableGraph *trainable_graph() const
  {
    assert(_tdata);
    return _tdata->trainable_graph();
  }
  const TrainableContextData *trainable_ctx_data() const
  {
    assert(_tdata);
    return _tdata.get();
  }

  virtual ITensorRegistry *genTrainingTensors() = 0;

protected:
  std::unique_ptr<TrainableContextData> _tdata;

public:
  std::shared_ptr<backend::ITensorRegistry> grad_tensor_registry;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BACKEND_TRAIN_TRAINABLE_CONTEXT_H__
