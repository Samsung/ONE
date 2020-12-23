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

#ifndef __ONERT_BACKEND_BUILTIN_BACKEND_H__
#define __ONERT_BACKEND_BUILTIN_BACKEND_H__

#include "BackendContext.h"
#include "Config.h"
#include "KernelGenerator.h"
#include "TensorBuilder.h"
#include "Tensor.h"

#include <backend/Backend.h>

#include <memory>

namespace onert
{
namespace backend
{
namespace builtin
{

class Backend : public ::onert::backend::Backend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<onert::backend::BackendContext> newContext(ContextData &&data) const override
  {
    auto context = std::make_unique<BackendContext>(this, std::move(data));
    // ControlFlow backend may not build tensors for itself because the backend's operation uses
    // tensors of other baceknd instead
    // But the backend builds tensors in case of that the controlflow operation may have constant
    // input or that consecutive controflow operations exist. We have to make them not to be built
    // later
    // 1. Constant input
    //   These tensors cannot be dynamic tensor, so let's do it as follows:
    //   - always skip copying
    //   - if it is operation's input in child subgraph: register "use" as constant input of the
    //   operations in child subgraph
    //   - if it is child subgraph's output: register "use" as constant input of the operations
    //   using it
    // 2. Consecutive controflow operation's intermediate tensor
    //   These tensors can be dynamic tensor and this is complicated to support without copying. But
    //   there is no such case until now, let's support it later
    // TODO Remove TensorBuilder and ConstantInitializer
    // TODO Support Consecutive controflow operation's intermediate tensor
    auto tr = std::make_shared<TensorRegistry>();
    auto tb = std::make_shared<TensorBuilder>(tr);
    context->tensor_registry = tr;
    context->tensor_builder = tb;
    context->kernel_gen = std::make_shared<KernelGenerator>(
      *context->graph(), tb->dynamicTensorManager(), tr, context->external_context());
    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_BACKEND_H__
