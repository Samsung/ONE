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

#ifndef __ONERT_BACKEND_TRAIN_BACKEND_CONTEXT_H__
#define __ONERT_BACKEND_TRAIN_BACKEND_CONTEXT_H__

#include <backend/train/TrainableBackendContext.h>

#include "ExternalContext.h"
#include "KernelGenerator.h"
#include "TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace train
{

// TODO Remove this class if ExecutorFactory creates trainable context only once instead of
// replacing BackendContext
class DummyBackendContext : public backend::BackendContext
{
public:
  DummyBackendContext(const Backend *backend, ContextData &&data,
                      std::shared_ptr<backend::ITensorRegistry> tensor_registry = nullptr)
    : backend::BackendContext(backend, std::move(data), tensor_registry)
  {
  }

  backend::ITensorRegistry *genTensors() override { return nullptr; }

  backend::FunctionMap genKernels() override { return backend::FunctionMap{}; }
};

// TODO Unify TensorBuilder
// TODO Unify TensorRegistry
class BackendContext : public onert::backend::train::TrainableBackendContext
{
public:
  BackendContext(const ITrainableBackend *backend, std::unique_ptr<TrainableContextData> &&tdata,
                 std::shared_ptr<backend::train::ITensorRegistry> tensor_registry = nullptr,
                 std::shared_ptr<TensorBuilder> tensor_builder = nullptr,
                 std::unique_ptr<exec::train::optimizer::Optimizer> optimizer = nullptr,
                 std::shared_ptr<KernelGenerator> kernel_gen = nullptr)
    : onert::backend::train::TrainableBackendContext(backend, std::move(tdata), tensor_registry),
      kernel_gen{kernel_gen}, _external_context(new ExternalContext),
      _tensor_builder{tensor_builder}, _optimizer{std::move(optimizer)}
  {
  }
  BackendContext(const BackendContext &) = delete;
  ~BackendContext() = default;

public:
  BackendContext &operator=(const BackendContext &) = delete;

public:
  backend::ITensorRegistry *genTensors() override;
  backend::train::ITensorRegistry *genTrainingTensors() override;

public:
  FunctionMap genKernels() override;

  std::shared_ptr<ExternalContext> external_context() { return _external_context; }

  const exec::train::optimizer::Optimizer *optimizer() const { return _optimizer.get(); }

private:
  FunctionMap generateFunctionMap();

public:
  // TODO Make it private
  std::shared_ptr<KernelGenerator> kernel_gen;

private:
  // NOTE ruy context has a thread pool, and when multiple ruy contexts are created,
  //      the thread pool is also created in duplicate
  // TODO Create one ruy context for session
  std::shared_ptr<ExternalContext> _external_context;

private:
  std::shared_ptr<TensorBuilder> _tensor_builder;

private:
  std::unique_ptr<exec::train::optimizer::Optimizer> _optimizer;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_BACKEND_CONTEXT_H__
