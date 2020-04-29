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

#ifndef __ONERT_BACKEND_CONTROLFLOW_BACKEND_H__
#define __ONERT_BACKEND_CONTROLFLOW_BACKEND_H__

#include "Config.h"
#include "KernelGenerator.h"

#include <backend/Backend.h>

#include <backend/IShapeFixer.h>

#include <memory>

namespace onert
{
namespace backend
{
namespace controlflow
{

class Backend : public ::onert::backend::Backend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<BackendContext> newContext(const ir::Graph &graph,
                                             const std::shared_ptr<custom::IKernelBuilder> &,
                                             bool) const override
  {
    const auto &operands = graph.operands();
    auto context = std::make_unique<BackendContext>(this, &graph);
    auto tb = std::shared_ptr<ITensorBuilder>(nullptr);
    context->tensor_builder = tb;
    context->constant_initializer = std::shared_ptr<IConstantInitializer>(nullptr);
    context->kernel_gen = std::make_shared<KernelGenerator>(operands);
    context->shape_fixer = std::shared_ptr<IShapeFixer>(std::make_shared<EmptyShapeFixer>());
    context->tensor_register = nullptr;
    context->optimizer = nullptr;
    return context;
  }

private:
  class EmptyShapeFixer : public IShapeFixer
  {
  public:
    EmptyShapeFixer() = default;

    void visit(const ir::operation::While &) override
    {
      // DO NOTHING
    }
  };

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_BACKEND_H__
