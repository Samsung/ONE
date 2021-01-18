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

#ifndef __ONERT_BACKEND_ACL_NEON_BACKEND_H__
#define __ONERT_BACKEND_ACL_NEON_BACKEND_H__

#include <memory>
#include <backend/Backend.h>
#include <ir/Operands.h>

#include "BackendContext.h"
#include "Config.h"
#include "ConstantInitializer.h"
#include "KernelGenerator.h"
#include "TensorManager.h"
#include "Optimizer.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{

class Backend : public ::onert::backend::Backend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<backend::BackendContext> newContext(ContextData &&data) const override
  {
    const auto &graph = *data.graph;
    const auto &operands = data.graph->operands();
    auto context = std::make_unique<acl_neon::BackendContext>(this, std::move(data));
    auto tm = createTensorManager(data.is_linear_executor);
    auto tr = std::make_shared<acl_common::AclTensorRegistry<TensorManager>>(tm);
    auto tb = std::make_shared<TensorBuilder>(operands, tm);
    context->tensor_registry = tr;
    context->tensor_builder = tb;
    context->constant_initializer = std::make_shared<ConstantInitializer>(operands, tr);
    context->kernel_gen = std::make_shared<KernelGenerator>(graph, tb, tr);
    context->optimizer = std::make_shared<Optimizer>(context.get());
    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace acl_neon
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_NEON_BACKEND_H__
