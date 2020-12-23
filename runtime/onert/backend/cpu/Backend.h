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

#ifndef __ONERT_BACKEND_CPU_BACKEND_H__
#define __ONERT_BACKEND_CPU_BACKEND_H__

#include "BackendContext.h"
#include "Config.h"
#include "KernelGenerator.h"

#include <backend/Backend.h>

#include <memory>

namespace onert
{
namespace backend
{
namespace cpu
{

class Backend : public ::onert::backend::Backend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<onert::backend::BackendContext> newContext(ContextData &&data) const override
  {
    auto custom_kernel_builder = data.custom_kernel_builder;
    auto &graph = *data.graph;
    auto context = std::make_unique<BackendContext>(this, std::move(data));
    auto tr = std::make_shared<cpu_common::TensorRegistry>();
    auto tb = std::make_shared<TensorBuilder>(tr);
    context->tensor_registry = tr;
    context->tensor_builder = tb;
    context->kernel_gen = std::make_shared<KernelGenerator>(graph, tb, tr, custom_kernel_builder,
                                                            context->external_context());
    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_BACKEND_H__
