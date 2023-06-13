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

#include <backend/Backend.h>
#include <backend/train/ITrainableBackend.h>

#include <memory>

namespace onert
{
namespace backend
{
namespace train
{

class Backend : public ::onert::backend::Backend, public ITrainableBackend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<TrainableBackendContext> newContext(TrainableContextData &&tdata) const override
  {
    const auto &tgraph = *tdata.tgraph;
    auto tr = std::make_shared<basic::TensorRegistry>();
    auto tb = std::make_shared<TensorBuilder>(tr, "Bump");
    auto grad_tr = std::make_shared<basic::TensorRegistry>();
    auto grad_tb = std::make_shared<TensorBuilder>(grad_tr, "Bump");
    auto tdata_ptr = std::make_unique<TrainableContextData>(std::move(tdata));
    auto context =
      std::make_unique<BackendContext>(this, std::move(tdata_ptr), tr, tb, grad_tr, grad_tb);

    // TODO Share tgraph, grad_tr, and grad_tb with KernelGenerator
    context->kernel_gen =
      std::make_shared<KernelGenerator>(tgraph, tr, grad_tr, context->external_context());

    return context;
  }

  // TODO Remove this constructor
  std::unique_ptr<onert::backend::BackendContext> newContext(ContextData &&data) const override
  {
    auto tr = std::make_shared<basic::TensorRegistry>();
    auto tb = std::make_shared<TensorBuilder>(tr, "Bump");
    auto context = std::make_unique<UntrainableBackendContext>(this, std::move(data), tr, tb);

    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_BACKEND_H__
