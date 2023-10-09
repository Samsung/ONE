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

#ifndef __ONERT_BACKEND_TRAIN_BACKEND_H__
#define __ONERT_BACKEND_TRAIN_BACKEND_H__

#include "BackendContext.h"
#include "Config.h"
#include "KernelGenerator.h"
#include "optimizer/Optimizers.h"

#include <backend/Backend.h>
#include <backend/train/ITrainableBackend.h>

#include <memory>

namespace onert
{
namespace backend
{
namespace train
{

// TODO Unify TensorBuilder
// TODO Unify TensorRegistry
class Backend : public ::onert::backend::Backend, public backend::train::ITrainableBackend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<onert::backend::BackendContext> newContext(ContextData &&data) const override
  {
    return std::make_unique<DummyBackendContext>(this, std::move(data));
  }

  std::unique_ptr<backend::train::TrainableBackendContext>
  newContext(backend::train::TrainableContextData &&tdata) const override
  {
    const auto &tgraph = *tdata.tgraph;
    auto optimizer = createOptimizer(tdata.optim_info);
    auto tr = std::make_shared<TensorRegistry>();
    auto tb = std::make_shared<TensorBuilder>(tr, optimizer.get(), "Bump");
    auto tdata_ptr = std::make_unique<backend::train::TrainableContextData>(std::move(tdata));
    auto context = std::make_unique<train::BackendContext>(this, std::move(tdata_ptr), tr, tb,
                                                           std::move(optimizer));

    context->kernel_gen = std::make_shared<train::KernelGenerator>(
      tgraph, tr, context->external_context(), context->optimizer());
    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_BACKEND_H__
