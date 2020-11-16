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
#include "ConstantInitializer.h"
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

  std::unique_ptr<onert::backend::BackendContext>
  newContext(const ir::Graph &graph, const std::shared_ptr<custom::IKernelBuilder> &kb,
             bool) const override
  {
    const auto &operands = graph.operands();
    const auto &operations = graph.operations();
    auto context = std::make_unique<BackendContext>(this, &graph);
    auto tr = std::make_shared<cpu_common::TensorRegistry>();
    auto tb = std::make_shared<TensorBuilder>(tr);
    context->tensor_registry = tr;
    context->tensor_builder = tb;
    context->constant_initializer = std::make_shared<ConstantInitializer>(operands, tr);
    context->kernel_gen = std::make_shared<KernelGenerator>(operands, operations, tb, tr, kb,
                                                            context->external_context());
    context->optimizer = nullptr;
    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_BACKEND_H__
